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

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

import logging
from typing import Dict, List, Set, Tuple

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.backend.base_backend import BaseLoRABackend, get_backend_from_name
from sglang.srt.lora.layers import BaseLayerWithLoRA, get_lora_layer
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import (
    LoRABatchInfo,
    LoRAType,
    get_customized_names_from_hf_names,
    get_layer_id,
    get_normalized_lora_weight_names,
    get_weight_name,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import replace_submodule

logger = logging.getLogger(__name__)


class LoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        lora_paths: Dict[str, str],
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        self.base_model: torch.nn.Module = base_model
        self.lora_paths: Dict[str, str] = lora_paths
        self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(lora_backend)

        self.init_loras()
        self.init_lora_memory_pool()
        
		# <NT> 老版实现方式：flashinfer里实现的分段矩阵乘法。
        # 里面会按分段后分别进行矩阵.
        # def run(
        #     self,
        #     x: torch.Tensor,
        #     weights: torch.Tensor,
        #     batch_size: int,
        #     weight_column_major: bool,
        #     seg_lens: Optional[torch.Tensor] = None,
        #     seg_indptr: Optional[torch.Tensor] = None,
        #     weight_indices: Optional[torch.Tensor] = None,
        # ) -> torch.Tensor:
        # x是A矩阵，weights是B矩阵，batch_size表示分多少片（一个seq一片），
        # weight_column_major表示B矩阵是否为列优先，对于fc层的gemm不是常规的gemm，B矩阵应该为列优先。
        # seg_lens，其实也是seq_len，是一个列表，里面每个元素表示该下标对应的片段长度（对应行数）。
        # seg_indptr，列表，表示每个seq的起始行号。
        # weight_indices，B矩阵每个分片的buffer id号。与 self.A_buffer[lora_weight_name][i][buffer_id] 中的[buffer_id]对应。

    def init_cuda_graph_batch_info(self, max_bs_in_cuda_graph: int):
        self.max_bs_in_cuda_graph = max_bs_in_cuda_graph
        with torch.device("cuda"):
            self.cuda_graph_batch_info = LoRABatchInfo(
                bs=self.max_bs_in_cuda_graph,
                seg_lens=torch.zeros(self.max_bs_in_cuda_graph, dtype=torch.int32),
                seg_indptr=torch.zeros(
                    self.max_bs_in_cuda_graph + 1, dtype=torch.int32
                ),
                max_len=0,
                weight_indices=torch.zeros(
                    self.max_bs_in_cuda_graph, dtype=torch.int32
                ),
                lora_ranks=torch.zeros(self.max_loras_per_batch, dtype=torch.int32),
                scalings=torch.zeros(self.max_loras_per_batch, dtype=torch.float),
            )

    # <NT> 1. 基于给定的lora_path，获取配置文件和目标module，configs和target_modules都是有多份的，对应不同的模块。
    #      2. 每个目标模块分配一个LoRAAdapter，并初始化权重 initialize_weights。self.loras的元素都是 LoRAAdapter 
    def init_loras(self):
        # Config of each LoRA adapter
        self.configs: Dict[str, LoRAConfig] = {}

        # Target module names in huggingface lora configs.
        # e.g., {"k_proj", "q_proj", "v_proj", "o_proj"}
        self.hf_target_names: Set[str] = set()
        for name, path in self.lora_paths.items():
            self.configs[name] = LoRAConfig(path)
            self.hf_target_names.update(self.configs[name].target_modules)

        # Target lora weight names for lora_a and lora_b modules respectively.
        weights_A: List[str] = []
        weights_B: List[str] = []
        for module in self.hf_target_names:
            lora_A, lora_B = get_normalized_lora_weight_names(module)
            weights_A += lora_A
            weights_B += lora_B
        self.lora_weight_names: Tuple[Set[str]] = set(weights_A), set(weights_B)

        # load all weights to cpu
        self.loras: Dict[str, LoRAAdapter] = {}
        for name in self.lora_paths.keys():
            lora_adapter = LoRAAdapter(
                name,
                self.configs[name],
                self.base_hf_config,
                self.load_config,
                self.lora_backend,
            )
            lora_adapter.initialize_weights()
            self.loras[name] = lora_adapter

        # misc lora configs
        self.max_lora_dim: int = max([x.hf_config["r"] for x in self.configs.values()])

        if self.lora_backend == "flashinfer":
            # FIXME remove the restrictions after supporting multi-rank for flashinfer backend
            max_lora_dim = max([x.hf_config["r"] for x in self.configs.values()])
            scaling = list(self.loras.values())[0].scaling
            assert all(x.hf_config["r"] == max_lora_dim for x in self.configs.values())
            assert all(x.scaling == scaling for x in self.loras.values())


        # <NT> monkey patch：即 “猴子补丁”，是一种在运行时动态修改代码的技术。
        # 将普通层替换层绑定了lora的层： set_lora_module-> get_lora_layer(如 MergedColumnParallelLinear 替换成 MergedColumnParallelLinearWithLoRA) / replace_submodule
        # 绑定了lora层的新层会先运行原来的层base_layer，然后额外运行lora部分计算。目前只有VocabParallelEmbedding和各种Linear层。
        # 把基础模型的对应层都替换成了lora层后，模型正常计算即可完成lora的计算。
        # 问：模型层已替换完成，如何跟seq绑定？
        # 答：看QKVParallelLinearWithLoRA的set_lora_info，ForwardBatch在构建的时候，如果服务初始化时有指定了lora（model_runner.server_args.lora_paths），会进而调用lora_manager.prepare_lora_batch。
        #     prepare_lora_batch 会判断 forward_batch.lora_paths是否存在，forward_batch.lora_paths是由req绑定并传入的，req有就有，没有就没有。
        #     如果req需要计算lora，会激活在服务初始化时就加载好的对应lora模块的计算数据，并进一步调用到 set_lora_info 确定需要计算lora，送入相应lora模块数据进行计算。
        # 
        # 总之，服务启动时，会设置要支持的lora模块，并完成加载和模型层的替换(普通层换成带lora的层)，但带lora的层不一定都会计算lora，
        #       需要看送入的req是否有指定lora模块（req与lora一一绑定，lora模块需要是在服务启动时设置的lora模块里所包含的），req有指定，则计算，没指定则不计算。
        # Convert original model layers to layers with LoRA
        self.convert_to_lora_layers()

    # <NT> lora内存池，里面包含A和B矩阵的内存分配。
    # 各个目标模块里的A矩阵如大小一致者，会复用内存。B矩阵亦然。
    def init_lora_memory_pool(self):
        # Initialize memory pool
        self.memory_pool = LoRAMemoryPool(
            self.base_hf_config,
            self.max_loras_per_batch,
            self.max_lora_dim,
            self.dtype,
            self.tp_size,
            self.tp_rank,
            self.lora_modules,
        )

        # Initialize target lora modules in memory pool
        self.memory_pool.init_buffers(self.lora_weight_names, self.base_model)

    # <NT> lora batch，即是多个lora模块组合而成的一个batch。因为每个lora模块里，针对相同的类型和层id，其权重维度都一样。
    # 可以把这些权重放到一起使用segment_gemm来一起计算。
    # prepare_lora_batch是通过server_args.lora_paths判断调用的，但具体该batch是否需要计算lora，需要cur_uids不为空。
    # cur_uids首先由forward_batch.lora_paths指定，是通过 req.lora_path -> ScheduleBatch.get_model_worker_batch -> ModelWorkerBatch -> ForwardBatch得到的。
    # 即 cur_uids 与req绑定，req有指定就有，没指定就不计算。
    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_paths)
        assert len(cur_uids) <= self.max_loras_per_batch
        self.memory_pool.prepare_lora_batch(cur_uids, self.loras)

        # set up batch info shared by all lora modules
        bs = forward_batch.batch_size

        if (
            hasattr(self, "max_bs_in_cuda_graph")
            and bs <= self.max_bs_in_cuda_graph
            and forward_batch.forward_mode.is_cuda_graph()
        ):
            # Do in-place updates when CUDA graph is enabled and the batch forward mode
            # could use CUDA graph.
            self.cuda_graph_batch_info.bs = bs
            self.cuda_graph_batch_info.seg_lens[:bs].fill_(1)
            torch.cumsum(
                self.cuda_graph_batch_info.seg_lens[:bs],
                dim=0,
                out=self.cuda_graph_batch_info.seg_indptr[1 : bs + 1],
            )
            self.cuda_graph_batch_info.max_len = 1

            for i, lora_path in enumerate(forward_batch.lora_paths):
                self.cuda_graph_batch_info.weight_indices[i] = (
                    self.memory_pool.get_buffer_id(lora_path)
                )
                if lora_path is not None:
                    lora = self.loras[lora_path]
                    self.cuda_graph_batch_info.lora_ranks[
                        self.cuda_graph_batch_info.weight_indices[i]
                    ] = lora.config.hf_config["r"]
                    self.cuda_graph_batch_info.scalings[
                        self.cuda_graph_batch_info.weight_indices[i]
                    ] = lora.scaling
            batch_info = self.cuda_graph_batch_info
        else:
            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.forward_mode.is_extend()
                else torch.ones(bs, device=self.device)
            )
            seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)
            seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
            max_len = int(torch.max(seg_lens))
            weight_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)

            lora_ranks = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.int64, device="cuda"
            )
            scalings = torch.zeros(
                (self.max_loras_per_batch,), dtype=torch.float, device="cuda"
            )
            for i, lora_path in enumerate(forward_batch.lora_paths):
                weight_indices[i] = self.memory_pool.get_buffer_id(lora_path)
                if lora_path is not None:
                    lora = self.loras[lora_path]
                    lora_ranks[weight_indices[i]] = lora.config.hf_config["r"]
                    scalings[weight_indices[i]] = lora.scaling
            batch_info = LoRABatchInfo(
                bs=bs,
                seg_lens=seg_lens,
                seg_indptr=seg_indptr,
                max_len=max_len,
                weight_indices=weight_indices,
                lora_ranks=lora_ranks,
                scalings=scalings,
            )
        self.lora_backend.set_batch_info(batch_info)

        # call set_lora_info for each lora modules
        # <NT> ab buffer都是按lora模块，层类型，层id三个索引进行划分。对于同类型和同id，不同lora模块的数据可以一起算。
        # 如 self.A_buffer[lora_weight_name][i][buffer_id]， 里面的[buffer_id]会跟lora模块一一对应。
        # 所以计算时，seg_indptr分a矩阵，weight_indices分b矩阵。各自单独计算gemm。搜 self.segment_gemm = SegmentGEMMWrapper(workspace_buffer) 看参数定义
        for layer_id, modules in self.lora_modules.items():
            for module_name, module in modules:
                if "qkv_proj" in module_name:
                    module.set_lora_info(
                        self.memory_pool.get_tensor(
                            "qkv_proj", layer_id, LoRAType.LORA_A
                        ),
                        self.memory_pool.get_tensor(
                            "q_proj", layer_id, LoRAType.LORA_B
                        ),
                        self.memory_pool.get_tensor(
                            "kv_proj", layer_id, LoRAType.LORA_B
                        ),
                    )
                else:
                    weight_name = get_weight_name(
                        module_name, self.lora_weight_names, LoRAType.LORA_A
                    )
                    module.set_lora_info(
                        self.memory_pool.get_tensor(
                            weight_name, layer_id, LoRAType.LORA_A
                        ),
                        self.memory_pool.get_tensor(
                            weight_name, layer_id, LoRAType.LORA_B
                        ),
                    )

    def set_lora_module(self, module_name, module):
        lora_module = get_lora_layer(module, self.lora_backend)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def convert_to_lora_layers(self):
        # Target module names of customized layers defined in python/sglang/srt/layers
        # e.g., {"qkv_proj", "o_proj"}
        customized_target_names = get_customized_names_from_hf_names(
            self.hf_target_names, self.base_model
        )

        # Monkey patch to use the LoRA version layers
        self.lora_modules: Dict[int, List[Tuple[str, BaseLayerWithLoRA]]] = {
            i: [] for i in range(self.base_hf_config.num_hidden_layers)
        }

        for module_name, module in self.base_model.named_modules():
            # TODO (lifuhuang): in the future, we should consider generalizing the
            # should_apply_lora function to support mapping by full module name instead
            # of just the last part (e.g., "qkv_proj") to support scenarios with multiple
            # attention stacks (e.g., multimodal models).
            # See: https://github.com/sgl-project/sglang/issues/6608
            if getattr(
                self.base_model, "should_apply_lora", None
            ) and not self.base_model.should_apply_lora(module_name):
                continue

            # The module should be converted if it is included in target_names
            if module_name.split(".")[-1] in customized_target_names:
                layer_id = get_layer_id(module_name)
                self.lora_modules[layer_id].append(
                    (module_name, self.set_lora_module(module_name, module))
                )
