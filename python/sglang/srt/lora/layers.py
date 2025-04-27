from typing import List, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora.backend.base_backend import BaseLoRABackend


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
        )

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        if self.lora_backend.fuse_stacked_lora_b:
            # TODO: avoid using contiguous() in GPU.
            # B_buffer_gate_up: (num_lora, 2 * output_dim, r)
            self.B_buffer_gate_up = torch.cat(
                (B_buffer[0], B_buffer[1]), dim=-2
            ).contiguous()
        else:
            self.B_buffer_gate_up = (B_buffer[0], B_buffer[1])

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}

        lora_output = self.lora_backend.run_gate_up_lora(
            x,
            self.A_buffer_gate_up,
            self.B_buffer_gate_up,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
        )

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

	# <NT> 该函数会在ForwardBatch.init_new->prepare_lora_batch中被调用，即每次推理前都会调用一次，以更新当前batch数据对应的lora信息
    # 信息重点是bs，seg_indptr（请求划分） 和 weight_indices。
    # set_lora是这个ForwardBatch是否需要计算lora的标志位，即如果调用到set_lora_info，则需要计算lora，否则不计算。看父类ColumnParallelLinearWithLoRA的forward函数。
    # 因为在初始化时，init_loras已经完成了对该模型的层的替换，把相对应的层都换成了带lora的新层。新层会先计算原始层的内容，然后计算lora。
    # 但因为不是每个请求都需要计算，而且lora模块会根据请求类型更换，所以有set_lora表示是否需要计算lora，其他参数表示要计算的lora内容。TODO:目前set_lora_info是
    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_q: torch.Tensor,
        B_buffer_kv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv

        if self.lora_backend.fuse_stacked_lora_b:
            assert (
                B_buffer_q.shape[-1] == B_buffer_kv.shape[-1]
            ), "The lora rank of q and kv should be the same when enabling fusion of qkv lora_b"
            output_dim_q, output_dim_kv = B_buffer_q.shape[-2], B_buffer_kv.shape[-2]

            # B_buffer_qkv: (num_lora, output_dim_q + 2 * output_dim_kv, r)
            self.B_buffer_qkv = torch.cat(
                (B_buffer_q[0], B_buffer_kv[0], B_buffer_kv[1]), dim=-2
            ).contiguous()

            # Offsets of q/k/v in output dimension
            self.output_offset = torch.tensor(
                [
                    0,
                    output_dim_q,
                    output_dim_q + output_dim_kv,
                    output_dim_q + 2 * output_dim_kv,
                ],
                dtype=torch.int32,
                device=B_buffer_q.device,
            )
            # For computing number of launched blocks
            self.max_qkv_out_dim = max(output_dim_q, output_dim_kv)
        else:
            self.B_buffer_qkv = (
                B_buffer_q,
                B_buffer_kv,
            )

    # <NT> apply_lora会在每次调用该层的forward时调用。
    # lora的计算公式：无lora Y=X*W, 有lora Y=X*(W+AB)
    # 可以分成两种方式计算：
    # 1. 事先将lora的AB矩阵合入到原来矩阵权重中，得到新的W1=W0+A*B，Y=X*W1
    # 2. 不保留修改后的权重，原来权重的计算照旧，即Y=X*W0 + X*A*B
    # 这里采用的是第二种方式，原因在于可以随着batch的数据，动态切换各种各样的lora模块，lora模块会跟batch里的每个seq一一对应。
    # (TokenizerManager) -> generate_request -> _tokenize_one_request -> (Scheduler) -> recv_from_tokenizer -> TokenizedGenerateReqInput
    #  -> recv_req.lora_path -> ModelWorkerBatch(python/sglang/srt/managers/schedule_batch.py#1106) -> ForwardBatch.lora_paths
    # 从最初的generate_request就已经有指定lora_path了。每个req对应一个lora_path，但是基础模型都共用一个，所以不采用动态执行而不保存修改后的权重。
    #
    # apply_lora函数里，base_output是基础模型的计算输出，即上面的X*W0, 是已经算好了的。
    # apply_lora函数需要计算X*A*B，然后与X*W0相加得到lora的叠加输出。
    # 注：当前版本已经融合成了run_qkv_lora，进一步阅读。
    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}
        if self.lora_backend.fuse_stacked_lora_b:
            backend_kwargs["output_offset"] = self.output_offset
            backend_kwargs["max_qkv_out_dim"] = self.max_qkv_out_dim

        lora_output = self.lora_backend.run_qkv_lora(
            x,
            self.A_buffer_qkv,
            self.B_buffer_qkv,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
        )

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(
        self, B: List[torch.Tensor], tp_rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B_q, B_kv = B
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        return B_q[q_start_idx:q_end_idx, :], B_kv[:, kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
        )

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B


def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
	# <NT> 下面是lora支持的层，左边是并入lora前的(即原始模型定义的)，右边是并入lora后的。
    # 函数输入的layer就是原始模型定义层，通过这个层，看能否找到对应的lora层，
    # 找到则构造一个对应lora层对象，并返回。如果找不到则不支持。
    # note: 都带有Parallel字样，因为这些层都考虑了张量并行，会按rank进行权重的加载和计算。
    #       lora的AB矩阵也要随之加载和分块计算。因为lora权重少，直接全量加载，计算时分块即可？
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
