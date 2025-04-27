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

# <NT> ���������Բ�offload��cpu�ķ���
# cuda graph�Ĳ������п��Բ���cpu��gpu�Ŀ�����Ҳ���Բ����gpu��cpu�Ŀ��������������ó��첽���� non_blocking���첽�������cpu���Եģ����Ὣ����뵽��ǰcuda stream�С�
# �����Ҫ���ڿ������������cpu�ļ��㣬��Ҫ��cpu����Ҳ���뵽��cuda stream�У���Ҫʹ��cudaLaunchHostFunc������
# ������ self.input_cpu.copy_(x, non_blocking=True)�� ����x��gpu��tensor���ݣ�ע�ⲻҪд��x.cpu(), ��Ϊx.cpu���漰����tensor��gpu��cpu�Ŀ������ڴ洴����
# ��cuda graph�������У����������ڴ�����Ĳ����������ڴ����������Ҫ�ڲ�����������С�
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
        # <NT> nn.Linear��Ȩ����(out_features, in_features), һ��in_features��Ԫ�����ڴ���������
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # <NT> ע��cuda graph�����޷�ִ��pythonԭ����print����Ϊgraph replay���涼��Ҫ��cuda������
        #      ��ʹ�� if torch.cuda.is_current_stream_capturing() �����Ĳ��֣�ֻ����capture�׶λ��ӡ��replay�׶ν������ӡ��
        #      ���Ǹ���cuda�汾��printf�����һ��kernel����ӡ����cpu��ӡ�Ƶ�cuda stream�
        # <NT> �൱�� return torch.matmul(x, layer.weight.t()) + bias��
        # ���ڸ�linear�ļ���ʵ�ֵ����Բ㣬������ʵ����ײ㶼�ǵ������pytorch��linearʵ�֡�
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

    # <NT> quant_method��quant_config�õ���quant_config���ģ�Ͷ�����ʱ����
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

# <NT> ������TP���е���ͨ���Բ�?
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

# <NT> Ȩ�ؾ����н��л��֣�����������ͨ����˷���
# Pytorch���nn.Linear��tf���tf.keras.layers.Dense����FC�㣬������������Ȩ��Tensor����gemm���㣬����Ҫ��Ȩ����ת�á�
# ��������з���Ե�����ͨ����˷�����ת�ú��Ȩ�����зֵġ�
# ע�⣡����: RowParallelLinear���з֣�ָ��GEMM��B������У���Ӧ��nn.Linear��Ȩ�ؾ������з֡���֮ColumnParallelLinear��Ȼ��
#
# GEMM�У�����X��һ�У���Ҫ��Ȩ��A��һ���ÿ��Ԫ��һһ��Ӧ��˲��ۼӵõ�һ��Ԫ�ؽ����
# ����Ȩ��A�����з֣���һ���豸��X��һ�к�A��һ���Ѿ�����ۼӼ��㣬�õ�һ��������ս�������ֻ��Ҫ��������������ռ��������ɣ���all gather��
# ���������У���Ϊÿ��Ԫ���Ѿ��������꣬�������Ĳ�Ҳ��Ҫ��TP�����Ը�����Ĭ�ϲ���all gather����gather_output=False��
# ColumnParallelLinear��ÿ���ڵ������������Ҫ�������ģ��������ϴ󣬻�Ƚ�ռ�Դ档
#
# ���ӣ�
#  ����������[0,1], Ȩ��[a,b]��������Ӧ����[0*a+1*c, 0*b+1*d]
#           [2,3]��    [c,d]               [2*a+3*c, 2*b+3*d]
#  ColumnParallelLinear������ʾ����Ҫ�������������ݡ�
#     ���з�ʱ���豸0�ֵ�һ��Ȩ��[a]�������豸b�ֵ�Ȩ��[b]. a�����[0*a+1*c], b�����[0*b+1*d], all gatherֱ��ƴ�Ӽ��õ����ս����
#                              [c]                  [d]        [2*a+3*c]        [2*b+3*d]
#  RowParallelLinear����input_is_parallel=Trueʱ����ǰ������з�û��all gather��������ʾ��
#     �豸a������[0], �豸b������[1], �豸aȨ����[a,b], �豸bȨ����[c,d], a�Ľ����[0*a, 0*b], b�Ľ����[1*c, 1*d], ��Ҫall reduce���ӵõ����ս����
#               [2]            [3]                                             [2*a, 2*b]          [3*c, 3*d],

# <NT> ע��˵��һά�����룬�ڶ�ά���������nn.Linear�е���(out_features, in_features), ��֮�෴����Ҫע�⡣
#      ���������з��ǰ�����A����ת�õ���������˷�����ģ�in_features��GEMM��K��
#      ��: �����ڹ���weightʱ������(out_features, in_features)�ģ�ʲôʱ������ת���ˣ�
#      ��: ����ǰ��û��ת�ã����Լ�����ColumnParallelLinear�����з֣�ʵ����GEMM���У�nn.linear���С�����ʵ�ʵ�nn.Linear��Ȩ��ʵ���������з֡�
#          ���仰˵ColumnParallelLinear�Ƕ�out_features���з֣�RowParallelLinear�Ƕ�in_features���з֡�
#          �������ӿ�����RowParallelLinear��weight_loader����ע�͡�
#          �������nn.LinearȨ�س䵱gemm��Bʱ��Ҫô��ת�ã�Ҫô��B����Ϊ�����ȵķ�ʽ���С�
#          (out_features, in_features)���Ϊ�����ȣ����൱�������ȵ�(in_features, out_features).
#      �ʣ����з�ΪʲôҪ��������δ���зֵġ�
#      ����Ϊ�������Բ�֮�����һ�������������������ԣ�����Ҫ������Ӧ���������Ĳ����зֵġ�
#          ����зּ���ֵX�����ڵ������Ҫ���ӣ���Y = GeLU(X1A1 + X2A2)���������GELU�Ƿ����Ե�GeLU(X1A1+X2A2) ��= GeLU(X1A1)+GeLU(X2A2)��
#          �������Ҫ�з�X����Ҫ��ִ��all-reduce�����ܽ��뼤�����
#          ������з�X������ֱ�ӽ��뼤�����ͬʱ����Ҫִ��all-gather����Ϊ�в���Ҫ���������зֺõģ�ֻ��Ҫ���в��к����һ��all-reduce���ɡ�
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
    
# <NT> ��������Բ��Ȩ�ؾ���ϲ�Ϊһ������ľ����ٰ��н��зָ
# �ϲ������Բ㴦��ƽ��������һ������ǰ��������ϵ��
# ����ϲ����ٸ�����output_sizes: List[int]��List�ж��ٸ�Ԫ�ؾ��Ǻϲ����ٸ���
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

# <NT> ���Բ�(ȫ���Ӳ�)������attention��QKV�����Ա任����Ȩ�ؾ����������ά��ƴ����һ������������QKV��ƴ�Ӿ���
# �ò������������л�����headά�Ȳ��У������key/value��head��query��headҪ��ʱ����MQA��GQA��
# ��ʱ������������Ҫ�з�headʱ��key/value��head�ᱻ���ƶ�ݣ�����һ�����ظ����㡣
# 
# ÿ�� head �����Կ�����һ����Զ�����������ȡ�������ǴӲ�ͬ�ĽǶȶ��������ݽ��д�����ȡ��ͬ������������� head �зֲ����ƻ�ģ�͵�����ṹ�͹��ܡ�
# MHA��k/v head��q head����һ�£�MQA�����и�q��Ӧһ��k/v��GQA�Ǽ���qΪһ�飬һ��q��Ӧһ��k/v�������������֣�k��v��head����һ�µģ�
#
# QKVParallelLinear�����в��ж���ƣ��в��������ͨgemm���ԣ�mk * kn = mn����kn�зֳɶ��kn0��kn1, kn2�ȡ�
# ��qkv��Ӧ����Ҳ�ǰ����ά��ƴ����һ��ģ���nά�ȣ�
# ��GQAΪ����total_num_heads=4��total_num_kv_heads=2����weight���£�
#  n      w_q       w_k    w_v
# k   00-00-00-00  11-11  22-22     
#     00-00-00-00  11-11  22-22
#     00-00-00-00  11-11  22-22
#     00-00-00-00  11-11  22-22
# �����hidden_state��Ӧ������������ֱ����Բ���㣬�ֱ�õ�q/k/v��
# Ȩ�ؾ���weight��nά�������ά�ȣ��õ���q/k/v�����������Ȩ�ؾ��������n��һ�µģ�����weight��n������ǰ�head���Ų��ġ�
# ���Եõ���q/k/v��������ʾ���������һ�е�h0h0���Ӧw_q�е�һ�е�00��
#  n         q                k        v
# m   h0h0-h1h1-h2h2-h3h3  h4h4-h5h5  h6h6-h7h7     
#     h0h0-h1h1-h2h2-h3h3  h4h4-h5h5  h6h6-h7h7
# ��Ϊ�ǻ����в��У���ÿ���ڵ�����������������ģ��һᰴhead�з� (���ά��n)��
# ��ڵ�0����w_q��h0��h1���֣�w_k��h4��w_v��h6, 
#   �ڵ�1����w_q��h2��h2���֣�w_k��h5��w_v��h7
# head�з֣������ڵ�������㣬kvcacheҲ�ǰ�head���ж�ȡ [layer_num][token_num][head_num][head_dim]
#
# �ʣ�Ϊʲô˵����ӦMQA��GQA����qͷ����k/vͷ���������£�������Ҫ��k/v head���ƶ�ݡ�
# ����Ϊһ��kv head��Ӧ���q head������qh0-qh3��Ӧkvh0��qh4-qh7��Ӧkvh1��
# ����ڵ�0���ֵ�����qh0-qh1����Ҫ��kvh0�����ڵ�1���ֵ�qh2-qh3��ҲҪ��kvh0����kv_head��ิ��һ�ݡ�
# ע�⣺����ķ�����Զ�ʹ��divide()�������У��������ݶ������ܱ�������
#      ���Բ��������kv_head������4���ڵ�����6���������������Ḵ�Ӳ��٣������߼��������á�
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
        # <NT> ֱ�Ӱ�q_head�����䣬ע���õĶ���divide��������ж��ܷ�����������Ĵ��붼����divide������Ҫ�����ܱ������ġ�
        self.num_heads = divide(self.total_num_heads, tp_size)
        # <NT> ������4��kv_head, ÿ��kv_head��Ӧ4��q_head, ��16��q_head.
        #      �����4���ڵ㣬��ÿ���ڵ�һ��һ����qҲ�պ�������, ÿ���ڵ���4��q_head�������num_kv_heads��num_kv_head_replicas��Ϊ1.
        #      �����6���ڵ�, �޷���q_head��������֧�֡�
        #      �����8���ڵ㣬�����num_heads��total_num_heads//tp_size=16//8=2, ��һ���ڵ���2��q_head, 01 23 45 67 89 1011 1213 1415, num_kv_headsΪ1��num_kv_head_replicas�����2��
        #                    �ڵ�(q_head, kv_head)-> 0(01,0) 1(23,0) 2(45,1) 3(67,1) 4(89,2) 5(1011,2) 6(1213,3) 7(1415,3)����kv_head���������ظ��ģ���Ӧnum_kv_head_replicas==2��
        #      �����2���ڵ㣬��num_kv_heads���Ա����䵽ÿ���ڵ�2����num_kv_head_replicas��Ȼ��1���ڵ��٣�����Ҫ���ơ�
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * self.head_size
        self.kv_proj_shard_size = self.num_kv_heads * self.head_size
        input_size = self.hidden_size
        # <NT> self.head_size��q_head��kv_head�����õģ�������head��head_size��һ�¡���ӦȨ�����ά��N.
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

    # <NT> �׵�ַ��q��k���׵�ַ��q����ƫ��q�Ĵ�С(self.num_heads * self.head_size), v���׵�ַ��k����ƫ���Լ��Ĵ�С(self.num_kv_heads * self.head_size)
    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + 2 * self.num_kv_heads) * self.head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    # <NT> ���ڷ����attention(��GQA), num_heads�����num_kv_heads�������䱶����ϵ�������������attention��num_heads��num_kv_headsһ��һ�¡�
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

# <NT> Ȩ�ؾ����н��л��֣���ͬ���豸�������Ȩ�ؾ���Ĳ�ͬ��������Ĳ��ֳ˻���
# �����Բ�����У���������水gemm�����㣬��Ҫ��nn.Linear�е�Ȩ����ת�á�ת�ú��Ȩ�زŻ�������Xһ�л��Ȩ��Aһ�У������ۼӵõ�һ���㡣
# ��������зֺ����з֣���Ե���ת�ú��Ȩ��A��˵�ģ�
# ���Խ�A���л��֣���Xһ�к�Aһ�м���õ�����һ�е���м��������ս������Ҫ�������м���Ľ���ۼ�������������Ҫall reduce���Ա�ColumnParallelLinear��
# ���������У���Ϊÿ��Ԫ�ض�û�����꣬����Ĳ�Ҫ��TP�������Ƚ������all reduce���ܣ����ܼ����㡣���Ը�����Ĭ��Ҫ��all reduce����reduce_results=True��
# input_is_parallel��ʾ���������Ѿ��зֺ��ˣ�����ǰ�����Դ���ColumnParallelLinearʹ�á���ColumnParallelLinear����Ҫall gather��RowParallelLinearҲ����Ҫ���з��������ݡ�
#
# ����ʱ�Ĳ���input_size��output_size������AȨ��ά����[output_size, input_size], X����ά����[batch_size, input_size], 
# Y���ά����[batch_size, output_size], ����ʱ��Ҫ��Ȩ��ת��(ֱ�ӵ���pytorch����ʱ����Ҫ)��
#
# ע�⣡����: RowParallelLinear���з֣�ָ��GEMM��B������У���Ӧ��nn.Linear��Ȩ�ؾ������з֡���֮ColumnParallelLinear��Ȼ��
#
# ���ӣ�
#  ����������[ 0, 1, 2, 3], Ȩ��[a,b,c,d]��������Ӧ����[ 0a+ 1e+ 2i+ 3m,  0b+ 1f+ 2j+ 3n,  0c+ 1g+ 2k+ 3o,  0d+ 1h+ 2l+ 3p]
#           [ 4, 5, 6, 7]��    [e,f,g,h]               [ 4a+ 5e+ 6i+ 7m,  4b+ 5f+ 6j+ 7n,  4c+ 5g+ 6k+ 7o,  4d+ 5h+ 6l+ 7p]
#           [ 8, 9,10,11]      [i,j,k,l]               [ 8a+ 9e+10i+11m,  8b+ 9f+10j+11n,  8c+ 9g+10k+11o,  8d+ 9h+10l+11p]
#           [12,13,14,15]      [m,n,o,p]               [12a+13e+14i+15m, 12b+13f+14j+15n, 12c+13g+14k+15o, 12d+13h+14l+15p]
#   
#  ColumnParallelLinear������ʾ����Ҫ�������������ݡ�
#     ���з�ʱ���豸0�ֵ�һ��Ȩ��[a,b]�������豸b�ֵ�Ȩ��[c,d]. a�����[ 0a+ 1e+ 2i+ 3m,  0b+ 1f+ 2j+ 3n], b�����[ 0c+ 1g+ 2k+ 3o,  0d+ 1h+ 2l+ 3p], all gatherֱ��ƴ�Ӽ��õ����ս����
#                              [e,f]                 [g,h]        [ 4a+ 5e+ 6i+ 7m,  4b+ 5f+ 6j+ 7n]         [ 4c+ 5g+ 6k+ 7o,  4d+ 5h+ 6l+ 7p]
#                              [i,j]                 [k,l]        [ 8a+ 9e+10i+11m,  8b+ 9f+10j+11n]         [ 8c+ 9g+10k+11o,  8d+ 9h+10l+11p]
#                              [m,n]                 [o,p]        [12a+13e+14i+15m, 12b+13f+14j+15n]         [12c+13g+14k+15o, 12d+13h+14l+15p]
#
#  RowParallelLinear����input_is_parallel=Trueʱ����ǰ������з�û��all gather������������ʾ��
#     �豸a������[ 0, 1], �豸b������[ 2, 3], �豸aȨ����[a,b,c,d], �豸bȨ����[i,j,k,l], a�Ľ����[  0a+1e,   0b+1f,   0c+1g,   0d+1h], b�Ľ����[  2i+3m,   2j+3n,   2k+3o,   2l+3p], ��Ҫall reduce���ӵõ����ս����
#               [ 4, 5]            [ 6, 7]            [e,f,g,h]             [m,n,o,p]          [  4a+5e,   4b+5f,   4c+5g,   4d+5h]          [  6i+7m,   6j+7n,   6k+7o,   6l+7p],
#               [ 8, 9]            [10,11]                                                     [  8a+9e,   8b+9f,   8c+9g,   8d+9h]          [10i+11m, 10j+11n, 10k+11o, 10l+11p]
#               [12,13]            [14,15]                                                     [12a+13e, 12b+13f, 12c+13g, 12d+13h]          [14i+15m, 14j+15n, 14k+15o, 14l+15p]
#
# 
## ˼�����⣬��һ����֣��׶��Խ���ͨ�ţ��ﵽ������ͨ��overlap��Ŀ�ġ�
# һ���в������Բ��ÿ���ڵ���������һ������batch�����ݵĲ����У�������RowParallelLinear���ӷ����������밴batchά���з֣���a�зֳ�[0,1]��[ 8, 9], b�зֳ�[2,3]��[10,11]�� 
#                                                                                                                        [4,5]  [12,13]        [6,,7]  [14,15]
#   a0��aȨ�ؼ���õ�[  0a+1e,   0b+1f,   0c+1g,   0d+1h],  b1��bȨ�ؼ���õ� [  2i+3m,   2j+3n,   2k+3o,   2l+3p],  allreduce�õ� [ 0a+ 1e+ 2i+ 3m,  0b+ 1f+ 2j+ 3n,  0c+ 1g+ 2k+ 3o,  0d+ 1h+ 2l+ 3p]
#                   [  4a+5e,   4b+5f,   4c+5g,   4d+5h]                    [  6i+7m,   6j+7n,   6k+7o,   6l+7p]                [ 4a+ 5e+ 6i+ 7m,  4b+ 5f+ 6j+ 7n,  4c+ 5g+ 6k+ 7o,  4d+ 5h+ 6l+ 7p]
#   a1�밡Ȩ�ؼ���õ� [ 8a+9e,   8b+9f,   8c+9g,   8d+9h]����������ͬ��allreduce�õ��°�� [ 8a+ 9e+10i+11m,  8b+ 9f+10j+11n,  8c+ 9g+10k+11o,  8d+ 9h+10l+11p]
#                    [12a+13e, 12b+13f, 12c+13g, 12d+13h]                               [12a+13e+14i+15m, 12b+13f+14j+15n, 12c+13g+14k+15o, 12d+13h+14l+15p]
#   �ڵ���Ա���ƴ�ӵõ����ս����
#   ���ݽ�һ���зֺ��������ݱ����tensor�б�ÿ��tensor��Ȩ��(����Ҫ�з�)����ͨ���Բ���㣬�������allreduce��ͬʱ������һ��tensor��
#
# �������batch_size��С��gemm�����ô��ܼ��ͷ��򿿣�����Ч�ʻή�ͣ�����batch_size�����С������²����˼����з֡�������������з֣��Ƿ���У�
#   ������RowParallelLinear���ӷ����������밴��ά���з֣���a�зֳ� [0] �� [1], b�зֳ� [2] �� [3]��
#                                                              [4]    [5]         [6]    [7]
#                                                              [8]    [9]        [10]   [11]
#                                                             [12]   [13]        [11]   [15]
#   ��ʱ����ά����[4,1]��Ȩ��ά����[2,4], �޷�ֱ��ʹ����ͨ�ľ���˷�����Ҫ��һ����Ȩ���з����һ���зֵõ�2��[1,4]����, aȨ���зֳ�[a,b,c,d]��[e,f,g,h], bȨ����[i,j,k,l]��[m,n,o,p]��
#   a0��a0Ȩ�ؼ���õ�[ 0a, 0b, 0c, 0d], b0��b0Ȩ�صõ�[ 2i, 2j, 2k, 2l], allreduce
#                    [ 4a, 4b, 4c, 4d]               [ 6i, 6j, 6k, 6l]
#                    [ 8a, 8b, 8c, 8d]               [10i,10j,10k,10l]
#                    [12a,12b,12c,12d]               [11i,11j,11k,11l]
#   a1��a1Ȩ�ؼ���õ�[ 1e, 1f, 1g, 1h] ������                          ��allreduce
#                    [ 5e, 5f, 5g, 5h]
#                    [ 9e, 9f, 9g, 9h]
#                    [13e,13f,13g,13h]
#   �ڵ��ڸ����ٵ���һ�Ρ����Եõ���ȷ���������allreduce��ͨ�������࣬��Ϊmk x kn = mn��ͨ������mn������зֵ�ά����k������ͨ����������mn��ͨ������ԭ����������ֻ�ܰ�ά��m�з֣�m/2��һ��ͨ��������mn/2������ͨ�����벻�з�һ�¡�
#
# ������n������Ȩ���Ƿ���У�
#     �豸a������[ 0, 1], �豸b������[ 2, 3], �豸aȨ����[a,b] [c,d], �豸bȨ����[i,j][k,l],
#               [ 4, 5]            [ 6, 7]            [e,f] [g,h]            [m,n][o,p],
#               [ 8, 9]            [10,11]                                             
#               [12,13]            [14,15]
#     a��a0Ȩ�ؼ���õ�[  0a+1e,   0b+1f], b��b0Ȩ�ؼ���õ�[  2i+3m,   2j+3n], allreduce �õ������ǰ���С�a��a1Ȩ�أ��Լ�b��b1Ȩ�ؼ��㣬���Եõ������С�
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
            # <NT> loaded_weight.narrow(dim, start, length), dim��Ҫ�зֵ�ά��, start����dimά���Ͽ�ʼ��Ƭ����ʼ������length��dimά���Ͽ�ʼ��Ƭ�ĳ��ȡ�
            # RowParallelLinear�İ����з������GEMM��B����ʽ�������зֵģ���(in_features, out_features)���У��зֺ�ÿ���豸�õ�(n, out_features)�����ݡ�
            # Ҳ���ǰ�����ά�ȷ�������з֣�����Ƭά����(in_features, out_features) -> (length, out_features)
            # 
            # input_dim�������ά���±꣬��nn.Linear��Ȩ��ά����(out_features, in_features)�������˵��෴����Ҫע�⡣
            # ����param_data(out_features, in_features)��˵Ҫ�з�����ά�ȣ�param_data.shape[input_dim]������豸�ᱻ�ֵ������ݵ�����ά�ȣ�
            # shard_size����sub_in_features, start_idx��tp_rank���䣬�õ���ά����(out_features, sub_in_features).
            # 
            # ע��: RowParallelLinear���з֣�ָ��GEMM��B������У���Ӧ��nn.Linear��Ȩ�ؾ������з֡���֮ColumnParallelLinear��Ȼ��
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
        # <NT> ���ǰ��ӵ���ColumnParallelLinear����ִ�е���Ĭ�ϲ���all gather���������Ѿ����зֺõ��ˣ�
        # ������Ҫ����split_tensor_along_last_dim���з�һ�¡�
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
        # <NT> Ĭ����Ҫ��Լ�����ӻ��ܽ��
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
