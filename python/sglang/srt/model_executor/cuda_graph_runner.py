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
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

import torch
import tqdm

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import get_available_gpu_memory, is_hip

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
                setattr(sub, "is_torch_compile", False)
            else:
                # NOTE: Temporarily workaround MoE
                if "FusedMoE" in sub.__class__.__name__:
                    if num_tokens == 1:
                        # The performance of torch.compile on this layer is not always good when bs > 1,
                        # so we decide to only use torch.compile when bs =1
                        sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
                setattr(sub, "is_torch_compile", True)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


# <NT> @contextmanager 是装饰符，用于简化上下文管理，它能够在进入和离开with语句块时自动执行一些代码。
# 例如，文件操作中，我们经常使用with语句来确保文件在使用后能正确关闭，这就是利用了文件对象的上下文管理器功能。
# 
# patch_model是根据是否需要做torch.compile, 而做了一层适配的中转，返回推理函数。
# torch.compile会把model.forward的内容固化成静态graph，会采用采用算子融合、内存布局优化等策略，提高计算效率。
# cuda graph是把cuda操作固化，如kernel launch和copy等，通过捕获（capture）一系列操作来减少启动开销。
#           仅用于decode，因为cuda graph虽然会加速launch的速度，但需要固定内存布局，需要按每个batch_size去固化graph，有一定显存开销。
#           在prefill阶段，launch损耗比例小，大头在计算，且bacth_size变化多样，难以兼顾，所以没必要为了省这点launch开销而浪费其他资源（如显存）。
# 二者不冲突，可一起使用。cuda graph可以使用torch.compile前或compile后的计算过程。
@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=False,
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024


def get_batch_sizes_to_capture(model_runner: ModelRunner):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs

    if capture_bs is None:
        if server_args.speculative_algorithm is None:
            if server_args.disable_cuda_graph_padding:
                capture_bs = list(range(1, 33)) + [64, 96, 128, 160]
            else:
                capture_bs = [1, 2, 4] + [i * 8 for i in range(1, 21)]
        else:
            # Since speculative decoding requires more cuda graph memory, we
            # capture less.
            capture_bs = list(range(1, 9)) + list(range(9, 33, 2)) + [64, 96, 128, 160]

    if _is_hip:
        capture_bs += [i * 8 for i in range(21, 33)]

    if max(capture_bs) > model_runner.req_to_token_pool.size:
        # In some case (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [model_runner.req_to_token_pool.size - 1] + [
            model_runner.req_to_token_pool.size
        ]

    capture_bs = list(sorted(set(capture_bs)))
    # <NT> 过滤掉预设数据里超过范围的部分
    capture_bs = [
        bs
        for bs in capture_bs
        if bs <= model_runner.req_to_token_pool.size
        and bs <= server_args.cuda_graph_max_bs
    ]
    # <NT> 使用torch.compile的话，需要先对forward做了compile后提供给cuda graph。否则直接原始的forward函数即可。
    # torch.compile和cuda graph针对的优化点不同，二者可一起使用，共同优化。
    # torch.compile默认都使用, 除非超出数据量范围限制torch_compile_max_bs（可能是防止内存过大？）
    # 按bs来划分的目的是因为二者的优化都和输入数据量有关，数据量不一致，优化的结果也不一致。
    # 因为优化的初始化时离线进行的，后续不更改，所以按一个bs一份graph的方式进行。
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.enable_dp_attention = model_runner.server_args.enable_dp_attention
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_eagle():
            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen")
            else:
                self.capture_forward_mode = ForwardMode.TARGET_VERIFY
                self.num_tokens_per_bs = (
                    self.model_runner.server_args.speculative_num_draft_tokens
                )

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
        self.seq_len_fill_value = (
            self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        # FIXME(lsyin): leave it here for now, I don't know whether it is necessary
        self.encoder_len_fill_value = 0
        self.seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

		# <NT> graph的输入输出数据内存，用于cuda graph的capture时会与计算流程绑定在一起，计算时用的就是这些buffer的内存数据。
        # 内存按最大batch_size来开辟，实际使用时会按特定batch_size去取。
        # Graph inputs
        with torch.device("cuda"):
            self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            self.mrope_positions = torch.zeros((3, self.max_bs), dtype=torch.int64)

            # Speculative_inference
            if (
                model_runner.spec_algorithm.is_eagle3()
                and not model_runner.is_draft_worker
            ):
                self.hidden_states = torch.zeros(
                    (
                        self.max_num_token,
                        3 * self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
                self.model_runner.model.set_eagle3_layers_to_capture()
            elif model_runner.spec_algorithm.is_eagle():
                self.hidden_states = torch.zeros(
                    (self.max_num_token, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                )

            if self.is_encoder_decoder:
                # NOTE: encoder_lens can influence the full_text_row_masked_out_mask tensor when doing mixed batch
                self.encoder_lens = torch.full(
                    (self.max_bs,), self.encoder_len_fill_value, dtype=torch.int32
                )
            else:
                self.encoder_lens = None

            if self.enable_dp_attention:
                self.gathered_buffer = torch.zeros(
                    (
                        self.max_bs * self.dp_size * self.num_tokens_per_bs,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
                self.global_num_tokens_gpu = torch.zeros(
                    (self.dp_size,), dtype=torch.int32
                )

        # Capture
        try:
            with self.model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n"
                "Possible solutions:\n"
                "1. disable cuda graph by --disable-cuda-graph\n"
                "2. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "4. set --cuda-graph-max-bs to a smaller value (e.g., 32)\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    @contextmanager
    def model_capture_mode(self):
        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = True
        if hasattr(self.model_runner.token_to_kv_pool, "capture_mode"):
            self.model_runner.token_to_kv_pool.capture_mode = True

        yield

        if hasattr(self.model_runner.model, "capture_mode"):
            self.model_runner.model.capture_mode = False
        if hasattr(self.model_runner.token_to_kv_pool, "capture_mode"):
            self.model_runner.token_to_kv_pool.capture_mode = False

    def can_run(self, forward_batch: ForwardBatch):
        if self.enable_dp_attention:
            total_global_tokens = sum(forward_batch.global_num_tokens_cpu)

            is_bs_supported = forward_batch.can_run_dp_cuda_graph and (
                total_global_tokens in self.graphs
                if self.disable_padding
                else total_global_tokens <= self.max_bs
            )
        else:
            is_bs_supported = (
                forward_batch.batch_size in self.graphs
                if self.disable_padding
                else forward_batch.batch_size <= self.max_bs
            )

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        # <NT> 如果是decode_only是支持的，但如果是encoder_decoder模型，则需要forward_batch.encoder_lens里的元素全部都大于0才行。
        # 因为cuda graph无法处理mixed batch的情况？TODO why
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )
        return is_bs_supported and is_encoder_lens_supported

    def capture(self):
        with graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            avail_mem = get_available_gpu_memory(
                self.model_runner.device, self.model_runner.gpu_id, empty_cache=False
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for bs in capture_range:
                if get_tensor_model_parallel_rank() == 0:
                    avail_mem = get_available_gpu_memory(
                        self.model_runner.device,
                        self.model_runner.gpu_id,
                        empty_cache=False,
                    )
                    capture_range.set_description(
                        f"Capturing batches ({avail_mem=:.2f} GB)"
                    )

                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    (
                        graph,
                        output_buffers,
                    ) = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output_buffers

                # Save gemlite cache after each capture
                save_gemlite_cache()

    # <NT> 基于一个batch_size去构建一个cuda graph。
    # 因为 CUDA Graph 是基于固定的操作序列构建的，所以输入数据的形状和类型需要固定，所以会以batch_size划分。
    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = torch.cuda.CUDAGraph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = self.mrope_positions[:, :bs]

        if self.enable_dp_attention:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [
                        num_tokens // self.dp_size + (i < bs % self.dp_size)
                        for i in range(self.dp_size)
                    ],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_num_tokens = self.global_num_tokens_gpu
            gathered_buffer = self.gathered_buffer[:num_tokens]
        else:
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        # <NT> 构建指定batch_size的推理数据 ForwardBatch
        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
        )

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
 
        # <NT> 使用虚拟输入数据，结合特定batch_size，构架一个ForwardBatch，并跑一遍模型，得到输出。
        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None

            logits_output = forward(input_ids, forward_batch.positions, forward_batch)
            return logits_output.next_token_logits, logits_output.hidden_states

        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()

            run_once()

        global global_graph_memory_pool
        # <NT> torch.cuda.graph是一个上下文管理器（context-manager），
        # 它能够将 CUDA 相关的操作捕获到一个 CUDAGraph 类的对象当中，以便后续进行 replay 操作。
        # 这里表示捕获的是 out = run_once() 这个过程，后面调用replay时，相当于再次执行 out = run_once()。
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
            out = run_once()

		# <NT> 这里返回的是内存池的id号，可以在其他graph执行时被填入，共用内存池。
        global_graph_memory_pool = graph.pool()
        return graph, out


    # <NT> 初始化的时候完成capture，实际使用的时候调用replay执行capture好的流程
    #      这个大的replay函数包含了拷贝数据进去和补pad操作，实际的replay是按补好pad的数据按选定的graph来做replay。
    def recapture_if_needed(self, forward_batch: ForwardBatch):
        # If the capture_hidden_mode changes, we need to recapture the graph
        hidden_mode_from_spec_info = getattr(
            forward_batch.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
        )
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and self.capture_hidden_mode != CaptureHiddenMode.FULL
        ):
            self.capture_hidden_mode = CaptureHiddenMode.FULL
            self.capture()
        elif (
            forward_batch.capture_hidden_mode != CaptureHiddenMode.FULL
            and self.capture_hidden_mode != hidden_mode_from_spec_info
        ):
            self.capture_hidden_mode = hidden_mode_from_spec_info
            self.capture()

    def replay_prepare(self, forward_batch: ForwardBatch):
        self.recapture_if_needed(forward_batch)

		# <NT> raw_bs实际用于推理时凑的batch_size, 但已被capture的batch_size不一定会刚好包含它，所以需要对它做pad操作，将其补到已被capture的bs上。
        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.enable_dp_attention:
            index = bisect.bisect_left(
                self.capture_bs, sum(forward_batch.global_num_tokens_cpu)
            )
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        # bs将会是被capture好的
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(1)
            self.out_cache_loc.zero_()

        # <NT> 将实际数据拷贝到输入buffer中，在graph捕获的run_once里会从这些buffer里面取数据。
        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        if forward_batch.decode_seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(1)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.decode_seq_lens_cpu)

        if self.is_encoder_decoder:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)
        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_bs].copy_(forward_batch.mrope_positions)
        if self.enable_dp_attention:
            self.global_num_tokens_gpu.copy_(forward_batch.global_num_tokens_gpu)

        if hasattr(forward_batch.spec_info, "hidden_states"):
            self.hidden_states[:raw_num_token] = forward_batch.spec_info.hidden_states

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices,
            self.seq_lens,
            forward_batch.seq_lens_sum + (bs - raw_bs),
            self.encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu,
        )

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs

    def replay(
        self, forward_batch: ForwardBatch, skip_attn_backend_init: bool = False
    ) -> LogitsProcessorOutput:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

		# <NT> 按指定的bs拿到对应的graph做replay，replay的过程是 out = run_once()，
        # 其中的out已经被指向了self.output_buffers[bs]，所以从self.output_buffers[bs]里取数据即可。
        # input和output的内存是所有batch_size对应的graph都共享的。
        # Replay
        self.graphs[self.bs].replay()
        next_token_logits, hidden_states = self.output_buffers[self.bs]

        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits[: self.raw_num_token],
            hidden_states=(
                hidden_states[: self.raw_num_token]
                if hidden_states is not None
                else None
            ),
        )
        return logits_output

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_eagle():
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=torch.zeros(
                        (num_tokens * self.model_runner.model_config.context_len),
                        dtype=torch.bool,
                        device="cuda",
                    ),
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )

        return spec_info
