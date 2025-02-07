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
import re

import torch

from sglang.srt.lora.lora import LoRAAdapter, get_lora_layer
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_flashinfer_available, replace_submodule

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    from flashinfer import SegmentGEMMWrapper


def get_module_name(name):
    # Fallback solution of mapping from config module name to module name in model class.
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    return params_mapping.get(name, name)


def get_hidden_dim(module_name, config):
    # Fallback solution of get_hidden_dim for different modules
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    if module_name in ["q_proj", "o_proj", "qkv_proj"]:
        return config.hidden_size, config.hidden_size
    elif module_name in ["kv_proj"]:
        return config.hidden_size, config.hidden_size // (
            config.num_attention_heads // config.num_key_value_heads
        )
    elif module_name == "gate_up_proj":
        return config.hidden_size, config.intermediate_size
    elif module_name == "down_proj":
        return config.intermediate_size, config.hidden_size
    else:
        raise NotImplementedError()


def get_stacked_name(name):
    # origin name -> (name for A, name for B)
    params_mapping = {
        "q_proj": ("qkv_proj", "q_proj"),
        "k_proj": ("qkv_proj", "kv_proj"),
        "v_proj": ("qkv_proj", "kv_proj"),
        "gate_proj": ("gate_up_proj", "gate_up_proj"),
        "up_proj": ("gate_up_proj", "gate_up_proj"),
    }
    return params_mapping.get(name, (name, name))


def get_layer_id(name):
    match = re.search(r"layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


class LoRAManager:
    def __init__(
        self,
        base_model,
        lora_paths,
        base_hf_config,
        max_loras_per_batch,
        load_config,
        dtype,
    ):
        self.base_model = base_model
        self.lora_paths = lora_paths
        self.base_hf_config = base_hf_config
        self.max_loras_per_batch = max_loras_per_batch
        self.load_config = load_config
        self.dtype = dtype

        workspace_buffer = torch.empty(1 * 1024 * 1024, dtype=torch.int8, device="cuda")
        # flashinfer里实现的分段矩阵乘法。
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
        self.segment_gemm = SegmentGEMMWrapper(workspace_buffer)

        self.init_loras()
        self.init_lora_memory_pool()
        self.init_lora_batch()

    def match_target_modules(self, module_name):
        for target_module in self.target_modules:
            if module_name.split(".")[-1] == target_module:
                return True
        return False

    def get_target_modules(self):
        modules = []
        for module_name, module in self.base_model.named_modules():
            if self.match_target_modules(module_name):
                modules.append((module_name, module))
        return modules

    def set_lora_module(self, module_name, module):
        lora_module = get_lora_layer(
            module, self.segment_gemm, self.max_lora_dim, self.scaling
        )
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    # 1. 基于给定的lora_path，获取配置文件和目标module，configs和target_modules都是有多份的，对应不同的模块。
    # 2. 每个目标模块分配一个LoRAAdapter，并初始化权重 initialize_weights。self.loras的元素都是 LoRAAdapter 
    def init_loras(self):
        # get configs and target modules
        self.configs = {}
        self.origin_target_modules = set()
        for name, path in self.lora_paths.items():
            self.configs[name] = LoRAConfig(path)
            self.origin_target_modules = set(self.origin_target_modules) | set(
                self.configs[name].target_modules
            )
        if hasattr(self.base_model, "get_module_name"):
            self.target_modules = {
                self.base_model.get_module_name(module)
                for module in self.origin_target_modules
            }
        else:
            logger.warning(
                "WARNING: get_module_name() is not defined, "
                "which is used to map config module name to model implementation module name."
                "Use the default one, but please check if it is correct for your model."
            )
            self.target_modules = {
                get_module_name(module) for module in self.origin_target_modules
            }
        self.target_weights = set(
            [get_stacked_name(module) for module in self.origin_target_modules]
        )

        # load all weights to cpu
        self.loras = []
        self.lora_id = {}
        for name in self.lora_paths.keys():
            self.lora_id[name] = len(self.loras)
            self.loras.append(
                LoRAAdapter(
                    name, self.configs[name], self.base_hf_config, self.load_config
                )
            )
            self.loras[-1].initialize_weights()

        # 处理一些其他的配置信息
        # misc lora configs
        self.max_lora_dim = max([x.hf_config["r"] for x in self.configs.values()])
        self.scaling = self.loras[0].scaling
        # FIXME remove the restrictions
        assert all(x.hf_config["r"] == self.max_lora_dim for x in self.configs.values())
        assert all(x.scaling == self.scaling for x in self.loras)

        # monkey patch：即 “猴子补丁”，是一种在运行时动态修改代码的技术。
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
        # monkey patch to use the LoRA version
        self.lora_modules = []
        for module_name, module in self.get_target_modules():
            self.lora_modules.append(
                (module_name, self.set_lora_module(module_name, module))
            )

    # lora内存池，里面包含A和B矩阵的内存分配。
    # 各个目标模块里的A矩阵如大小一致者，会复用内存。B矩阵亦然。
    def init_lora_memory_pool(self):
        # preallocate lora memory pool
        self.A_buffer = {}
        self.B_buffer = {}
        num_layer = self.base_hf_config.num_hidden_layers
        for module_A, module_B in self.target_weights:
            # init A tensor, column_major=True
            if hasattr(self.base_model, "get_hidden_dim"):
                hidden_dim_A, _ = self.base_model.get_hidden_dim(module_A)
            else:
                logger.warning(
                    "WARNING: get_hidden_dim() is not defined, "
                    "which is used to get the hidden dim for different lora modules"
                    "Use the default one, but please check if it is correct for your model."
                )
                hidden_dim_A, _ = get_hidden_dim(module_A, self.base_hf_config)
            c = self.loras[-1].get_stacked_multiply(module_A)
            if module_A not in self.A_buffer:
                self.A_buffer[module_A] = [
                    torch.empty(
                        (
                            self.max_loras_per_batch,
                            self.max_lora_dim * c,
                            hidden_dim_A,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(num_layer)
                ]
            # init B tensor, column_major=True
            if hasattr(self.base_model, "get_hidden_dim"):
                _, hidden_dim_B = self.base_model.get_hidden_dim(module_B)
            else:
                logger.warning(
                    "WARNING: get_hidden_dim() is not defined, "
                    "which is used to get the hidden dim for different lora modules"
                    "Use the default one, but please check if it is correct for your model."
                )
                _, hidden_dim_B = get_hidden_dim(module_B, self.base_hf_config)
            c = self.loras[-1].get_stacked_multiply(module_B)
            if module_B not in self.B_buffer:
                self.B_buffer[module_B] = [
                    torch.empty(
                        (
                            self.max_loras_per_batch,
                            hidden_dim_B * c,
                            self.max_lora_dim,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(num_layer)
                ]

    def init_lora_batch(self):
        self.active_uids = set()  # set of active loras
        self.buffer_id = {}  # lora uid -> idx in memory pool

    def get_weight_name(self, name, idx):
        for target_weight_name in self.target_weights:
            if target_weight_name[idx] in name:
                return target_weight_name[idx]

    def load_lora(self, uid, buffer_id):
        num_layer = self.base_hf_config.num_hidden_layers
        if uid is None:
            for i in range(num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] *= 0
            return

        for i in range(num_layer):
            layer_weights = self.loras[self.lora_id[uid]].layers[i].weights # 一个lora模块里有多个层，每个层都有自己的名字和权重，[buffer_id]可对应lora模块id
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = self.get_weight_name(name, 0)
                    if lora_weight_name:
                        self.A_buffer[lora_weight_name][i][buffer_id].copy_(weights)
                else:
                    lora_weight_name = self.get_weight_name(name, 1)
                    if lora_weight_name:
                        self.B_buffer[lora_weight_name][i][buffer_id].copy_(weights)

    # lora batch，即是多个lora模块组合而成的一个batch。因为每个lora模块里，针对相同的类型和层id，其权重维度都一样。
    # 可以把这些权重放到一起使用segment_gemm来一起计算。
    # prepare_lora_batch是通过server_args.lora_paths判断调用的，但具体该batch是否需要计算lora，需要cur_uids不为空。
    # cur_uids首先由forward_batch.lora_paths指定，是通过 req.lora_path -> ScheduleBatch.get_model_worker_batch -> ModelWorkerBatch -> ForwardBatch得到的。
    # 即 cur_uids 与req绑定，req有指定就有，没指定就不计算。
    def prepare_lora_batch(self, forward_batch: ForwardBatch):
        # load active loras into lora memory pool
        cur_uids = set(forward_batch.lora_paths)
        assert len(cur_uids) <= self.max_loras_per_batch
        i = 0
        j = len(self.active_uids)
        evictable_uids = list(self.active_uids)
        for uid in cur_uids:
            if uid not in self.active_uids:
                if j < self.max_loras_per_batch:
                    index = j
                    j += 1
                else:
                    while i < len(evictable_uids) and evictable_uids[i] in cur_uids:
                        i += 1
                    assert i < len(evictable_uids)
                    self.active_uids.remove(evictable_uids[i])
                    self.buffer_id.pop(evictable_uids[i])
                    index = i
                    i += 1
                self.load_lora(uid, index)
                self.active_uids.add(uid)
                self.buffer_id[uid] = index

        # 这里有个return，如不需要计算lora，会从这里跳出去
        if cur_uids == set([None]):
            return

        # setup lora in forward modules
        bs = forward_batch.batch_size
        seg_lens = (
            forward_batch.extend_seq_lens
            if forward_batch.forward_mode.is_extend()
            else torch.ones(bs, device="cuda")
        )
        # FIXME: reuse the data rather than recompute
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        weight_indices = torch.empty((bs,), dtype=torch.int64, device="cuda")
        for i, lora_path in enumerate(forward_batch.lora_paths):
            weight_indices[i] = self.buffer_id[lora_path]

        for module_name, module in self.lora_modules:
            layer_id = get_layer_id(module_name)

            # ab buffer都是按lora模块，层类型，层id三个索引进行划分。对于同类型和同id，不同lora模块的数据可以一起算。
            # 如 self.A_buffer[lora_weight_name][i][buffer_id]， 里面的[buffer_id]会跟lora模块一一对应。
            # 所以计算时，seg_indptr分a矩阵，weight_indices分b矩阵。各自单独计算gemm。搜 self.segment_gemm = SegmentGEMMWrapper(workspace_buffer) 看参数定义
            if "qkv_proj" not in module_name:
                weight_name = self.get_weight_name(module_name, 0)
                module.set_lora_info(
                    self.A_buffer[weight_name][layer_id],
                    self.B_buffer[weight_name][layer_id],
                    bs,
                    seg_indptr,
                    weight_indices,
                )
            else:
                module.set_lora_info(
                    self.A_buffer["qkv_proj"][layer_id],
                    self.B_buffer["q_proj"][layer_id],
                    self.B_buffer["kv_proj"][layer_id],
                    bs,
                    seg_indptr,
                    weight_indices,
                )
