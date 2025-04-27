"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/linear.py"""

import itertools
import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
    _ColumnvLLMParameter,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

WEIGHT_LOADER_V2_SUPPORTED = [
    "CompressedTensorsLinearMethod",
    "AWQMarlinLinearMethod",
    "AWQLinearMethod",
    "GPTQMarlinLinearMethod",
    "Fp8LinearMethod",
    "BlockInt8LinearMethod",
    "MarlinLinearMethod",
    "QQQLinearMethod",
    "GPTQMarlin24LinearMethod",
    "TPUInt8LinearMethod",
    "GPTQLinearMethod",
    "FBGEMMFp8LinearMethod",
    "ModelOptFp8LinearMethod",
    "ModelOptFp4LinearMethod",
    "IPEXAWQLinearMethod",
]


def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_bitsandbytes_4bit_shard(
    param: Parameter, shard_offsets: Dict[str, Tuple[int, int]], loaded_shard_id: str
) -> Tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""

    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param[shard_id], loaded_weight


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

# <NT> 分析将线性层offload的cpu的方案
# cuda graph的捕获流中可以捕获cpu到gpu的拷贝，也可以捕获从gpu到cpu的拷贝，但必须设置成异步拷贝 non_blocking，异步是相对于cpu而言的，但会将其插入到当前cuda stream中。
# 如果需要基于拷贝后的数据做cpu的计算，需要将cpu计算也插入到该cuda stream中，需要使用cudaLaunchHostFunc函数。
# 另外如 self.input_cpu.copy_(x, non_blocking=True)， 其中x是gpu的tensor数据，注意不要写成x.cpu(), 因为x.cpu会涉及自身tensor的gpu到cpu的拷贝和内存创建，
# 在cuda graph捕获流中，不允许有内存申请的操作，所有内存申请操作需要在捕获流外面进行。
class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        # <NT> nn.Linear的权重是(out_features, in_features), 一行in_features个元素在内存上连续。
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # <NT> 注意cuda graph里面无法执行python原生的print，因为graph replay里面都需要是cuda操作。
        #      如使用 if torch.cuda.is_current_stream_capturing() 包含的部分，只能在capture阶段会打印，replay阶段将不会打印。
        #      除非改用cuda版本的printf，如编一个kernel做打印。或将cpu打印推到cuda stream里。
        # <NT> 相当于 return torch.matmul(x, layer.weight.t()) + bias。
        # 对于该linear文件中实现的线性层，非量化实现最底层都是调用这个pytorch的linear实现。
        return F.linear(x, layer.weight, bias)


class LinearBase(torch.nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    # <NT> quant_method由quant_config得到，quant_config会从模型定义层的时候传入
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# <NT> 就是无TP并行的普通线性层?
class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
        )

        # All the linear layer supports quant method.
        assert self.quant_method is not None
        self.quant_method.create_weights(
            self,
            self.input_size,
            [self.output_size],
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        output = self.quant_method.apply(self, x, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s

# <NT> 权重矩阵按列进行划分，这里依照普通矩阵乘法。
# Pytorch里的nn.Linear和tf里的tf.keras.layers.Dense，即FC层，如果在外面调用权重Tensor进行gemm计算，是需要对权重做转置。
# 这里的列切分针对的是普通矩阵乘法，按转置后的权重来切分的。
# 注意！！！: RowParallelLinear行切分，指代GEMM中B矩阵的行，对应到nn.Linear的权重就是列切分。反之ColumnParallelLinear亦然。
#
# GEMM中，输入X的一行，需要和权重A的一列里，每个元素一一对应相乘并累加得到一个元素结果。
# 这里权重A按列切分，则一个设备中X的一行和A的一列已经完成累加计算，得到一个点的最终结果，因此只需要将其他点的数据收集起来即可，即all gather。
# 张量并行中，因为每个元素已经独立算完，进入后面的层也需要做TP，所以该算子默认不做all gather，即gather_output=False。
# ColumnParallelLinear的每个节点的输入数据需要是完整的，如果输入较大，会比较占显存。
#
# 例子：
#  假设有输入[0,1], 权重[a,b]，计算结果应该是[0*a+1*c, 0*b+1*d]
#           [2,3]，    [c,d]               [2*a+3*c, 2*b+3*d]
#  ColumnParallelLinear如下所示，需要完整的输入数据。
#     列切分时，设备0分到一列权重[a]，另外设备b分到权重[b]. a结果是[0*a+1*c], b结果是[0*b+1*d], all gather直接拼接即得到最终结果。
#                              [c]                  [d]        [2*a+3*c]        [2*b+3*d]
#  RowParallelLinear，令input_is_parallel=True时，即前面的列切分没做all gather，如下所示：
#     设备a输入有[0], 设备b输入有[1], 设备a权重有[a,b], 设备b权重有[c,d], a的结果是[0*a, 0*b], b的结果是[1*c, 1*d], 需要all reduce叠加得到最终结果。
#               [2]            [3]                                             [2*a, 2*b]          [3*c, 3*d],

# <NT> 注释说第一维是输入，第二维是输出。而nn.Linear中的是(out_features, in_features), 与之相反，需要注意。
#      所以这里切分是按矩阵A不做转置的正常矩阵乘法来算的，in_features是GEMM的K。
#      问: 但是在构建weight时数据是(out_features, in_features)的，什么时候做过转置了？
#      答: 加载前后都没做转置，可以简单理解成ColumnParallelLinear的列切分，实际是GEMM的列，nn.linear的行。对于实际的nn.Linear的权重实际上是行切分。
#          换句话说ColumnParallelLinear是对out_features做切分，RowParallelLinear是对in_features做切分。
#          具体例子看下面RowParallelLinear的weight_loader函数注释。
#          另外对于nn.Linear权重充当gemm的B时，要么先转置，要么以B矩阵为列优先的方式进行。
#          (out_features, in_features)标记为列优先，即相当于行优先的(in_features, out_features).
#      问：列切分为什么要求输入是未做切分的。
#      答：因为两个线性层之间存在一个激活函数，激活函数非线性，所以要求输入应该是完整的不做切分的。
#          如果切分激活值X，两节点输出需要叠加，则Y = GeLU(X1A1 + X2A2)，而激活函数GELU是非线性的GeLU(X1A1+X2A2) ！= GeLU(X1A1)+GeLU(X2A2)，
#          所以如果要切分X，需要先执行all-reduce，才能进入激活函数。
#          如果不切分X，可以直接进入激活函数，同时不需要执行all-gather，因为行并行要求输入是切分好的，只需要在行并行后面加一个all-reduce即可。
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        output_sizes: list of output sizes packed into one output, like for QKV
                       the list would be size 3.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        output_sizes: Optional[List[int]] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
    ):
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.gather_output = gather_output
        self.use_presharded_weights = use_presharded_weights

        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        assert self.quant_method is not None
        self.output_size_per_partition = divide(self.output_size, tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, tp_size) for output_size in self.output_sizes
            ]

        if output_sizes is None:
            output_sizes = [output_size]

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)

        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if output_dim is not None and not use_bitsandbytes_4bit:
            shard_size = param_data.shape[output_dim]
            start_idx = self.tp_rank * shard_size
            if not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: Parameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, _ColumnvLLMParameter):
            param.load_column_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # FIXME: This branch is needed to load deepseek v3 awq.
            # However, we should fix this and avoid the branching here.
            param.load_column_parallel_weight(loaded_weight)

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s
    
# <NT> 将多个线性层的权重矩阵合并为一个更大的矩阵，再按列进行分割。
# 合并的线性层处于平级，输入一样，无前后依赖关系。
# 具体合并多少个，看output_sizes: List[int]，List有多少个元素就是合并多少个。
class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
    ):
        self.output_sizes = output_sizes
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        assert all(output_size % tp_size == 0 for output_size in output_sizes)
        self.use_presharded_weights = use_presharded_weights
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=use_presharded_weights,
        )
        self.prefix = prefix

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.data[loaded_shard_id].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 2:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            shard_offsets: List[Tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id)
                    )

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                shard_size = loaded_weight.shape[output_dim]
                shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_offset = loaded_shard_id * shard_size
            param_data = param_data.narrow(0, shard_offset, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: List[Tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[int] = None,
    ):
        if loaded_shard_id is None:
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    shard_id=0,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(
                    loaded_weight=loaded_weight,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                )
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id < len(self.output_sizes)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = self.quant_method.quant_config.weight_block_size
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (
                (sum(self.output_sizes[:loaded_shard_id]) + block_n - 1) // block_n
            ) // self.tp_size
            shard_size = (
                (self.output_sizes[loaded_shard_id] + block_n - 1)
                // block_n
                // self.tp_size
            )
        else:
            shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
            shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            use_presharded_weights=self.use_presharded_weights,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )

# <NT> 线性层(全连接层)，用于attention的QKV的线性变换，其权重矩阵沿着输出维度拼接在一起，输出矩阵就是QKV的拼接矩阵。
# 该层在张量并行中会沿着head维度并行，但如果key/value的head比query的head要少时，如MQA和GQA，
# 此时做张量并行需要切分head时，key/value的head会被复制多份，会有一定的重复计算。
# 
# 每个 head 都可以看作是一个相对独立的特征提取器，它们从不同的角度对输入数据进行处理，提取不同方面的特征。按 head 切分不会破坏模型的整体结构和功能。
# MHA的k/v head与q head数量一致，MQA是所有个q对应一个k/v，GQA是几个q为一组，一组q对应一个k/v。（无论是哪种，k和v的head都是一致的）
#
# QKVParallelLinear基于列并行而设计，列并行针对普通gemm而言，mk * kn = mn，将kn切分成多个kn0，kn1, kn2等。
# 且qkv对应部分也是按输出维度拼接在一起的，即n维度，
# 以GQA为例，total_num_heads=4，total_num_kv_heads=2，则weight如下：
#  n      w_q       w_k    w_v
# k   00-00-00-00  11-11  22-22     
#     00-00-00-00  11-11  22-22
#     00-00-00-00  11-11  22-22
#     00-00-00-00  11-11  22-22
# 输入的hidden_state本应跟这三个矩阵分别线性层计算，分别得到q/k/v。
# 权重矩阵weight的n维度是输出维度，得到的q/k/v矩阵的列数与权重矩阵的列数n是一致的，所以weight的n方向就是按head来排布的。
# 所以得到的q/k/v将如下所示，里面如第一列的h0h0会对应w_q中第一列的00：
#  n         q                k        v
# m   h0h0-h1h1-h2h2-h3h3  h4h4-h5h5  h6h6-h7h7     
#     h0h0-h1h1-h2h2-h3h3  h4h4-h5h5  h6h6-h7h7
# 因为是基于列并行，即每个节点的输入数据是完整的，且会按head切分 (输出维度n)。
# 如节点0负责w_q的h0和h1部分，w_k的h4，w_v的h6, 
#   节点1负责w_q的h2和h2部分，w_k的h5，w_v的h7
# head切分，各个节点独立计算，kvcache也是按head进行读取 [layer_num][token_num][head_num][head_dim]
#
# 问：为什么说当对应MQA和GQA这种q头数比k/v头数多的情况下，可能需要将k/v head复制多份。
# 答：因为一个kv head对应多个q head，比如qh0-qh3对应kvh0，qh4-qh7对应kvh1，
# 如果节点0划分到数据qh0-qh1，就要有kvh0；而节点1划分到qh2-qh3，也要有kvh0。就kv_head会多复制一份。
# 注意：后面的分配策略都使用divide()函数进行，所有数据都必须能被整除。
#      所以不会出现如kv_head数量是4，节点数是6的情况，这样分配会复杂不少，下面逻辑不再适用。
class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        load_presharded_attn: bool = False,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        # <NT> 直接按q_head来分配，注意用的都是divide，里面会判断能否被整除，后面的代码都用了divide，即都要求是能被整除的。
        self.num_heads = divide(self.total_num_heads, tp_size)
        # <NT> 假设有4个kv_head, 每个kv_head对应4个q_head, 共16个q_head.
        #      如果有4个节点，则每个节点一人一个，q也刚好能整除, 每个节点有4个q_head。下面的num_kv_heads和num_kv_head_replicas都为1.
        #      如果有6个节点, 无法被q_head整除，不支持。
        #      如果有8个节点，上面的num_heads是total_num_heads//tp_size=16//8=2, 即一个节点有2个q_head, 01 23 45 67 89 1011 1213 1415, num_kv_heads为1，num_kv_head_replicas计算得2。
        #                    节点(q_head, kv_head)-> 0(01,0) 1(23,0) 2(45,1) 3(67,1) 4(89,2) 5(1011,2) 6(1213,3) 7(1415,3)，即kv_head会有两份重复的，对应num_kv_head_replicas==2。
        #      如果有2个节点，按num_kv_heads可以被分配到每个节点2个，num_kv_head_replicas自然是1，节点少，不需要复制。
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        input_size = self.hidden_size
        # <NT> self.head_size是q_head和kv_head都共用的，即所有head的head_size都一致。对应权重输出维度N.
        output_size = (
            (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        )
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]
        self.use_presharded_weights = load_presharded_attn

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_presharded_weights=self.use_presharded_weights,
        )

    # <NT> 首地址是q，k的首地址是q往后偏移q的大小(self.num_heads * self.head_size), v的首地址是k往后偏移自己的大小(self.num_kv_heads * self.head_size)
    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    # <NT> 对于分组的attention(如GQA), num_heads会大于num_kv_heads，且是其倍数关系。其他不分组的attention，num_heads和num_kv_heads一般一致。
    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,
            "k": self.num_kv_heads * self.head_size,
            "v": self.num_kv_heads * self.head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            if not self.use_presharded_weights:
                loaded_weight_shard = loaded_weight.narrow(
                    param.output_dim, shard_offset, shard_size
                )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = self.quant_method.quant_config.weight_block_size
            block_n, _ = weight_block_size[0], weight_block_size[1]
            shard_offset = (shard_offset + block_n - 1) // block_n
            shard_size = (shard_size + block_n - 1) // block_n

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
            use_presharded_weights=self.use_presharded_weights,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type and loaded_shard_id is not None:
            idx_map = {"q": 0, "k": 1, "v": 2}
            param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 3:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "total": (
                            (self.total_num_heads + 2 * self.total_num_kv_heads)
                            * self.head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                if not self.use_presharded_weights:
                    loaded_weight_shard = loaded_weight.narrow(
                        output_dim, shard_offset, shard_size
                    )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "total": (
                        (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            if loaded_shard_id == "q":
                shard_id = self.tp_rank
            else:
                shard_id = self.tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size

            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit and not self.use_presharded_weights:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size, shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

# <NT> 权重矩阵按行进行划分，不同的设备负责计算权重矩阵的不同行与输入的部分乘积。
# 在线性层计算中，如果在外面按gemm来计算，需要对nn.Linear中的权重做转置。转置后的权重才会有输入X一行会和权重A一列，计算累加得到一个点。
# 这里的行切分和列切分，针对的是转置后的权重A来说的，
# 所以将A按行划分，则X一行和A一行计算得到的是一行点的中间结果，最终结果还需要将其他行计算的结果累加起来，所以需要all reduce。对比ColumnParallelLinear。
# 张量并行中，因为每个元素都没有算完，后面的层要做TP，必须先将结果做all reduce汇总，才能继续算。所以该算子默认要做all reduce，即reduce_results=True。
# input_is_parallel表示输入数据已经切分好了，可以前面层可以搭配ColumnParallelLinear使用。则ColumnParallelLinear不需要all gather，RowParallelLinear也不需要再切分输入数据。
#
# 创建时的参数input_size和output_size，则其A权重维度是[output_size, input_size], X输入维度是[batch_size, input_size], 
# Y输出维度是[batch_size, output_size], 计算时需要将权重转置(直接调用pytorch计算时不需要)。
#
# 注意！！！: RowParallelLinear行切分，指代GEMM中B矩阵的行，对应到nn.Linear的权重就是列切分。反之ColumnParallelLinear亦然。
#
# 例子：
#  假设有输入[ 0, 1, 2, 3], 权重[a,b,c,d]，计算结果应该是[ 0a+ 1e+ 2i+ 3m,  0b+ 1f+ 2j+ 3n,  0c+ 1g+ 2k+ 3o,  0d+ 1h+ 2l+ 3p]
#           [ 4, 5, 6, 7]，    [e,f,g,h]               [ 4a+ 5e+ 6i+ 7m,  4b+ 5f+ 6j+ 7n,  4c+ 5g+ 6k+ 7o,  4d+ 5h+ 6l+ 7p]
#           [ 8, 9,10,11]      [i,j,k,l]               [ 8a+ 9e+10i+11m,  8b+ 9f+10j+11n,  8c+ 9g+10k+11o,  8d+ 9h+10l+11p]
#           [12,13,14,15]      [m,n,o,p]               [12a+13e+14i+15m, 12b+13f+14j+15n, 12c+13g+14k+15o, 12d+13h+14l+15p]
#   
#  ColumnParallelLinear如下所示，需要完整的输入数据。
#     列切分时，设备0分到一列权重[a,b]，另外设备b分到权重[c,d]. a结果是[ 0a+ 1e+ 2i+ 3m,  0b+ 1f+ 2j+ 3n], b结果是[ 0c+ 1g+ 2k+ 3o,  0d+ 1h+ 2l+ 3p], all gather直接拼接即得到最终结果。
#                              [e,f]                 [g,h]        [ 4a+ 5e+ 6i+ 7m,  4b+ 5f+ 6j+ 7n]         [ 4c+ 5g+ 6k+ 7o,  4d+ 5h+ 6l+ 7p]
#                              [i,j]                 [k,l]        [ 8a+ 9e+10i+11m,  8b+ 9f+10j+11n]         [ 8c+ 9g+10k+11o,  8d+ 9h+10l+11p]
#                              [m,n]                 [o,p]        [12a+13e+14i+15m, 12b+13f+14j+15n]         [12c+13g+14k+15o, 12d+13h+14l+15p]
#
#  RowParallelLinear，令input_is_parallel=True时，即前面的列切分没做all gather，做简化如下所示：
#     设备a输入有[ 0, 1], 设备b输入有[ 2, 3], 设备a权重有[a,b,c,d], 设备b权重有[i,j,k,l], a的结果是[  0a+1e,   0b+1f,   0c+1g,   0d+1h], b的结果是[  2i+3m,   2j+3n,   2k+3o,   2l+3p], 需要all reduce叠加得到最终结果。
#               [ 4, 5]            [ 6, 7]            [e,f,g,h]             [m,n,o,p]          [  4a+5e,   4b+5f,   4c+5g,   4d+5h]          [  6i+7m,   6j+7n,   6k+7o,   6l+7p],
#               [ 8, 9]            [10,11]                                                     [  8a+9e,   8b+9f,   8c+9g,   8d+9h]          [10i+11m, 10j+11n, 10k+11o, 10l+11p]
#               [12,13]            [14,15]                                                     [12a+13e, 12b+13f, 12c+13g, 12d+13h]          [14i+15m, 14j+15n, 14k+15o, 14l+15p]
#
# 
## 思考问题，进一步拆分，阶段性进行通信，达到计算与通信overlap的目的。
# 一，行并行线性层的每个节点的输入会有一个整个batch的数据的部分列，按上面RowParallelLinear例子分析，将输入按batch维度切分，将a切分成[0,1]和[ 8, 9], b切分成[2,3]和[10,11]。 
#                                                                                                                        [4,5]  [12,13]        [6,,7]  [14,15]
#   a0与a权重计算得到[  0a+1e,   0b+1f,   0c+1g,   0d+1h],  b1与b权重计算得到 [  2i+3m,   2j+3n,   2k+3o,   2l+3p],  allreduce得到 [ 0a+ 1e+ 2i+ 3m,  0b+ 1f+ 2j+ 3n,  0c+ 1g+ 2k+ 3o,  0d+ 1h+ 2l+ 3p]
#                   [  4a+5e,   4b+5f,   4c+5g,   4d+5h]                    [  6i+7m,   6j+7n,   6k+7o,   6l+7p]                [ 4a+ 5e+ 6i+ 7m,  4b+ 5f+ 6j+ 7n,  4c+ 5g+ 6k+ 7o,  4d+ 5h+ 6l+ 7p]
#   a1与啊权重计算得到 [ 8a+9e,   8b+9f,   8c+9g,   8d+9h]，。。。。同理allreduce得到下半段 [ 8a+ 9e+10i+11m,  8b+ 9f+10j+11n,  8c+ 9g+10k+11o,  8d+ 9h+10l+11p]
#                    [12a+13e, 12b+13f, 12c+13g, 12d+13h]                               [12a+13e+14i+15m, 12b+13f+14j+15n, 12c+13g+14k+15o, 12d+13h+14l+15p]
#   节点各自本地拼接得到最终结果。
#   数据进一步切分后，输入数据变成了tensor列表，每个tensor与权重(不需要切分)按普通线性层计算，将结果做allreduce的同时计算下一个tensor。
#
# 二，如果batch_size很小，gemm会往访存密集型方向靠，计算效率会降低，所以batch_size本身较小的情况下不适宜继续切分。如果继续按列切分，是否可行？
#   按上面RowParallelLinear例子分析，将输入按列维度切分，将a切分成 [0] 和 [1], b切分成 [2] 和 [3]。
#                                                              [4]    [5]         [6]    [7]
#                                                              [8]    [9]        [10]   [11]
#                                                             [12]   [13]        [11]   [15]
#   此时输入维度是[4,1]，权重维度是[2,4], 无法直接使用普通的矩阵乘法，需要进一步将权重行方向进一步切分得到2个[1,4]才行, a权重切分成[a,b,c,d]和[e,f,g,h], b权重有[i,j,k,l]和[m,n,o,p]。
#   a0和a0权重计算得到[ 0a, 0b, 0c, 0d], b0和b0权重得到[ 2i, 2j, 2k, 2l], allreduce
#                    [ 4a, 4b, 4c, 4d]               [ 6i, 6j, 6k, 6l]
#                    [ 8a, 8b, 8c, 8d]               [10i,10j,10k,10l]
#                    [12a,12b,12c,12d]               [11i,11j,11k,11l]
#   a1和a1权重计算得到[ 1e, 1f, 1g, 1h] 。。。                          ，allreduce
#                    [ 5e, 5f, 5g, 5h]
#                    [ 9e, 9f, 9g, 9h]
#                    [13e,13f,13g,13h]
#   节点内各自再叠加一次。可以得到正确结果，但是allreduce的通信量增多，因为mk x kn = mn，通信量是mn，如果切分的维度是k，两次通信量都还是mn，通信量是原来的两倍，只能按维度m切分，m/2，一次通信量就是mn/2，两次通信量与不切分一致。
#
# 三，从n方向切权重是否可行？
#     设备a输入有[ 0, 1], 设备b输入有[ 2, 3], 设备a权重有[a,b] [c,d], 设备b权重有[i,j][k,l],
#               [ 4, 5]            [ 6, 7]            [e,f] [g,h]            [m,n][o,p],
#               [ 8, 9]            [10,11]                                             
#               [12,13]            [14,15]
#     a与a0权重计算得到[  0a+1e,   0b+1f], b与b0权重计算得到[  2i+3m,   2j+3n], allreduce 得到结果的前两列。a与a1权重，以及b与b1权重计算，可以得到后两列。
#                     [  4a+5e,   4b+5f]                  [  6i+7m,   6j+7n]
#                     [  8a+9e,   8b+9f]                  [10i+11m, 10j+11n]
#                     [12a+13e, 12b+13f]                  [14i+15m, 14j+15n]
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_presharded_weights: bool = False,
    ):
        super().__init__(
            input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        # Divide the weight matrix along the last dimension.
        if tp_rank is None:
            tp_rank = get_tensor_model_parallel_rank()
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.input_size_per_partition = divide(input_size, self.tp_size)
        assert self.quant_method is not None
        self.use_presharded_weights = use_presharded_weights

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=[self.output_size],
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if (
            input_dim is not None
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            # <NT> loaded_weight.narrow(dim, start, length), dim是要切分的维度, start是在dim维度上开始切片的起始索引，length是dim维度上开始切片的长度。
            # RowParallelLinear的按行切分是针对GEMM的B矩阵方式来按行切分的，即(in_features, out_features)的行，切分后每个设备得到(n, out_features)的数据。
            # 也就是按输入维度方向进行切分，即切片维度是(in_features, out_features) -> (length, out_features)
            # 
            # input_dim是输入的维度下标，而nn.Linear的权重维度是(out_features, in_features)，与矩阵乘的相反，需要注意。
            # 对于param_data(out_features, in_features)来说要切分输入维度，param_data.shape[input_dim]是这该设备会被分到的数据的输入维度，
            # shard_size就是sub_in_features, start_idx按tp_rank分配，得到的维度是(out_features, sub_in_features).
            # 
            # 注意: RowParallelLinear行切分，指代GEMM中B矩阵的行，对应到nn.Linear的权重就是列切分。反之ColumnParallelLinear亦然。
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        if isinstance(param, RowvLLMParameter):
            # This `BasevLLMParameter` is defined in sglang/srt/layers/parameter.py,
            # It supports additional parameters like tp_rank and use_presharded_weights.
            param.load_row_parallel_weight(
                loaded_weight,
                tp_rank=self.tp_rank,
                use_presharded_weights=self.use_presharded_weights,
            )
        else:
            # `params` is defined in `vllm/model_executor/parameter.py`,
            # It does not support additional parameters.
            param.load_row_parallel_weight(loaded_weight)

    def forward(self, input_):
        # <NT> 如果前面接的是ColumnParallelLinear，且执行的是默认不做all gather，则其结果已经是切分好的了，
        # 否则需要调用split_tensor_along_last_dim，切分一下。
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias=bias_)
        # <NT> 默认需要规约，叠加汇总结果
        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        return output, output_bias

    def extra_repr(self) -> str:
        s = f"input_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
