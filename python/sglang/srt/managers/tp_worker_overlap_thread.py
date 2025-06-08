# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

import dataclasses
import logging
import signal
import threading
from queue import Queue
from typing import Optional, Tuple

import psutil
import torch

from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import DynamicGradMode, get_compiler_backend
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )

# <NT> overlap模式下，对TpModelWorker的进一步封装，里面会多开一个forward_thread，并绑定cuda stream，专门用于模型推理。
# 配有input_queue和output_queue. 主线程送数据到input_queue, 单独的forward_thread会持续读取input_queue，并进行推理。
class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(
            server_args, gpu_id, tp_rank, pp_rank, dp_rank, nccl_port
        )
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device
        self.gpu_id = gpu_id

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int64, device=self.device
        )

        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = torch.get_device_module(self.device).Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()
        self.scheduler_stream = torch.get_device_module(self.device).current_stream()
        if self.device == "cpu":
            self.scheduler_stream.synchronize = lambda: None  # No-op for CPU

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_group(self):
        return self.worker.get_tp_group()

    def get_attention_tp_group(self):
        return self.worker.get_attention_tp_group()

    def get_attention_tp_cpu_group(self):
        return self.worker.get_attention_tp_cpu_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def forward_thread_func(self):
        try:
            with torch.get_device_module(self.device).stream(self.forward_stream):
                self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error(f"TpModelWorkerClient hit an exception: {traceback}")
            self.parent_process.send_signal(signal.SIGQUIT)

    # <NT> cpu schedule 和 gpu compute overlap
    # 参考：https://zhuanlan.zhihu.com/p/17744625577
    # 普通模式：准备batch1 -> launch和计算batch1 -> 得到batch1生成的token1 -> 准备新的batch2 -> launch和计算batch2.
    # overlap模式：核心是future_token_ids, 表示当前在计算的batch的结果将会存放的位置（仅仅是地址索引，而不是实际内容）。
    #             准备batch需要较多cpu操作，但在下一次kernel实际计算时才需要知道要计算的token的实际内容，
    #             在准备batch阶段并不需要知道，只需要基于对用地址进行准备即可。
    #             
    #             self.future_token_ids_map: 维护维护实际生成的next_token_ids。
    #             future_token_ids_ct：一个batch的next_token_ids在map的偏移量，该偏移量会每个batch递增。
    #
    #             如batch1在启动计算的同时准备batch2的数据，正常流程是需要拿到batch1的结果next_token_ids去组建batch2的。
    #             这里则直接用future_token_ids_ct在map中划定一块区域，以该区域充当batch1的结果next_token_ids，完成batch2的组建，并launch batch2计算。
    #             batch1计算完成后，会有一个拷贝操作，将batch1的next_token_ids拷贝到map之前组batch2时划定的区域上进行填充。因为batch1计算 / 拷贝更新 / batch2计算
    #             都是gpu操作，都在cuda stream中完成，也就是会按顺序执行的。所以batch2计算时，batch2的输入会真正准备好。
    # 
    @DynamicGradMode()
    def forward_thread_func_(self):
        batch_pt = 0
        batch_lists = [None] * 2

        while True:
            # <NT> 从input_queue中拿到future_token的地址索引号在map中的偏移量，该偏移量会每个batch递增。
            model_worker_batch, future_token_ids_ct, sync_event = self.input_queue.get()
            if not model_worker_batch:
                break

            sync_event.wait()
            
            # <NT> 将model_worker_batch维系在batch_lists中，以免被释放。
            # Keep a reference of model_worker_batch by storing it into a list.
            # Otherwise, the tensor members of model_worker_batch will be released
            # by pytorch and cause CUDA illegal memory access errors.
            batch_lists[batch_pt % 2] = model_worker_batch
            batch_pt += 1

            # Create event
            copy_done = torch.get_device_module(self.device).Event()

            # Resolve future tokens in the input
            input_ids = model_worker_batch.input_ids
            resolve_future_token_ids(input_ids, self.future_token_ids_map)

            # Run forward
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.worker.forward_batch_generation(
                    model_worker_batch, model_worker_batch.launch_done
                )
            )

            # <NT> 当前batch1计算完，将输出下标更新到map的future_token_ids_ct偏移区域中，
            # 该区域已经被下一个batch2充当输入token_ids并launch kernel了。
            # batch1计算/map填充拷贝/batch2计算在cuda stream中排序，所以batch2在计算时，map肯定完成了填充，即输入数据能真正准备好。
            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            self.future_token_ids_map[
                future_token_ids_ct + 1 : future_token_ids_ct + bs + 1
            ] = next_token_ids

            # Copy results to the CPU
            if model_worker_batch.return_logprob:
                logits_output.next_token_logprobs = (
                    logits_output.next_token_logprobs.to("cpu", non_blocking=True)
                )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.to("cpu", non_blocking=True)
                    )
            if logits_output.hidden_states is not None:
                logits_output.hidden_states = logits_output.hidden_states.to(
                    "cpu", non_blocking=True
                )
            next_token_ids = next_token_ids.to("cpu", non_blocking=True)
            copy_done.record()

            # <NT> next_token_ids是真实输出的id号，用于取词后处理。
            self.output_queue.put(
                (copy_done, logits_output, next_token_ids, can_run_cuda_graph)
            )

    def resolve_last_batch_result(self, launch_done: Optional[threading.Event] = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        copy_done, logits_output, next_token_ids, can_run_cuda_graph = (
            self.output_queue.get()
        )

        if launch_done is not None:
            launch_done.wait()
        copy_done.synchronize()

        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = (
                logits_output.next_token_logprobs.tolist()
            )
            if logits_output.input_token_logprobs is not None:
                logits_output.input_token_logprobs = tuple(
                    logits_output.input_token_logprobs.tolist()
                )
        next_token_ids = next_token_ids.tolist()
        return logits_output, next_token_ids, can_run_cuda_graph

    def forward_batch_generation(
        self, model_worker_batch: ModelWorkerBatch
    ) -> Tuple[None, torch.Tensor, bool]:
        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )

        # A cuda stream sync here to avoid the cuda illegal memory access error.
        sync_event = torch.get_device_module(self.device).Event()
        sync_event.record(self.scheduler_stream)

		# <NT> 基于future_token_ids_ct号(表示未来输出token_id存放位置的内容偏移量)，去启动下一个batch的kernel。
        #      推送数据到input_queue中，在forward_thread中会while(True)循环获取input_queue的数据进行推计算(forward_thread_func_)
        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch, self.future_token_ids_ct, sync_event))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device=self.device,
        )
        # <NT> 随batch按batch_size大小递增。
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids, False

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.worker.update_weights_from_disk(recv_req)
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.worker.init_weights_update_group(recv_req)
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.worker.update_weights_from_distributed(recv_req)
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.worker.update_weights_from_tensor(recv_req)
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        return self.worker.get_weights_by_name(recv_req)

    def __delete__(self):
        self.input_queue.put((None, None))
        self.copy_queue.put((None, None, None))
