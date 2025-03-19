from abc import abstractmethod
from typing import List, Optional

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sglang.srt.utils import set_weight_attrs
from sglang.srt.layers.quantization.base_config import (
    QuantizeMethodBase,
)

from sglang.ext.host_launcher import host_launcher

# # GPTQ
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
##
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.model_executor.layers.quantization.gptq import GPTQConfig, ExllamaState
# #

OFFLOAD2CPU = False
OFFLOAD2CPU_DATAMODE = 0 # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO

# gptq���ɿӵ㣺1. gptq��zero��Ҫ����+1���ο�autogptq�Ĵ���ó���
#              2. gptq��Ȩ���Ƿ�ת�õģ�����Ҫ��linear����㣬��Ҫת�á���ִ��weight_t = weight.t(), Ȼ��ʹ��weight_t.data_ptr()���뵽cpp�У��ᷢ�ֵõ������ݲ�û��ת�ã�
#          

class MemoryPool:
    def __init__(self, len, dtype=torch.float32, device='cpu', pin_memory=False):
        self.pool = torch.zeros(len, dtype=dtype, device=device, pin_memory=pin_memory)
        self.used_index = 0

    def allocate(self, shape, is_from_head = False):
        if (is_from_head):
            self.used_index = 0

        num_elements = torch.prod(torch.tensor(shape)).item()
        if self.used_index + num_elements > self.pool.numel():
            print("out of memory: used {0}, need {1}, total {2} {3}".format(self.used_index, shape, self.pool.numel(), self.pool.device))
            return None

        sub_tensor = self.pool[self.used_index:self.used_index + num_elements]
        self.used_index += num_elements
        return sub_tensor.view(shape)

    def reset(self):
        self.used_index = 0

# ����ģʽ�����������״ε�����res=OffloadResource(1)ʱ�������ٴε������ûص�һ�ε�ʵ��
class LazySingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
class OffloadResource(metaclass=LazySingletonMeta):
    def __init__(self, mode):
        self.launcher = host_launcher.HostLauncher()
        config = host_launcher.linear.LinearConfig(mode) 
        # note: �䵱GPTQCpuLinearMethod��̬ʵ���������ͨ����ʱ���������ʱ��ס(��Ϊ��Դδ�ͷŶ��ȴ�������ŵ�self����������ģʽ��������������)
        self.cpu_linear = host_launcher.linear.Linear(config) 
        if (mode == 1):
            self.cpu_memory_pool = MemoryPool(len=2*172*17920, dtype=torch.float16, device='cpu', pin_memory=True)  # w4a16_fp16
            self.cuda_memory_pool = MemoryPool(len=172*17920, dtype=torch.float16, device='cuda')
        else:
            self.cpu_memory_pool = MemoryPool(len=2*172*17920, dtype=torch.bfloat16, device='cpu', pin_memory=True) # bf16
            self.cuda_memory_pool = MemoryPool(len=172*17920, dtype=torch.bfloat16, device='cuda')

    def get_launcher(self):
        return self.launcher
    
    def get_linear(self):
        return self.cpu_linear
    
    def get_memory_pool(self):
        return self.cpu_memory_pool, self.cuda_memory_pool

# ��Ӧfrom sglang.srt.layers.linear import LinearMethodBase
# ��дһ�ݷ�ֹimport�ɻ�
class CpuLinearMethodBase(QuantizeMethodBase):
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
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
    
# �Ա�UnquantizedLinearMethod
class UnquantizedCpuLinearMethod(CpuLinearMethodBase):
    LINEAR_ON_CPU_CNT = 0

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
        self.cpu_linear_idx = UnquantizedCpuLinearMethod.LINEAR_ON_CPU_CNT
        UnquantizedCpuLinearMethod.LINEAR_ON_CPU_CNT += 1 
        print("Create UnquantizedCpuLinearMethod", self.cpu_linear_idx, "on cpu")

        self.data_mode = OFFLOAD2CPU_DATAMODE
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
                device=torch.device('cpu')
            ),
            requires_grad=False,
        )

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # offload�����Դ
        g_res = OffloadResource(0)
        self.cpu_launcher = g_res.get_launcher()
        self.cpu_linear = g_res.get_linear()
        self.cpu_memory_pool, self.cuda_memory_pool = g_res.get_memory_pool()

        self.inputs_cpu = []
        self.outputs_cpu = []
        self.outputs_gpu = []

        for m in range(1, 173):
            sub_tensor = self.cpu_memory_pool.allocate((m, input_size_per_partition), True)
            self.inputs_cpu.append(sub_tensor)
            sub_tensor = self.cpu_memory_pool.allocate((m, sum(output_partition_sizes)), False)
            self.outputs_cpu.append(sub_tensor)
            sub_tensor = self.cuda_memory_pool.allocate((m, sum(output_partition_sizes)), True)
            self.outputs_gpu.append(sub_tensor)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO 
            idx = x.shape[0] - 1
            # print("apply cpu", idx, "stream", torch.cuda.current_stream().cuda_stream)
            self.inputs_cpu[idx].copy_(x, non_blocking=True)
            #######################################################
            self.cpu_launcher.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.cpu_linear.forward(
                    self.inputs_cpu[idx].shape[0],  # M
                    self.outputs_cpu[idx].shape[1], # N
                    self.inputs_cpu[idx].shape[1],  # K
                    self.inputs_cpu[idx].data_ptr(), 
                    layer.weight.data_ptr(),
                    0,
                    self.outputs_cpu[idx].data_ptr()
                )
            )
            self.outputs_gpu[idx].copy_(self.outputs_cpu[idx], non_blocking=True) #
            if bias is not None:
                self.outputs_gpu[idx].add_(bias)
            return self.outputs_gpu[idx]
        else:
            if (bias != None):
                output_cpu = F.linear(x.to("cpu"), layer.weight, bias.to("cpu"))
            else:
                output_cpu = F.linear(x.to("cpu"), layer.weight, None)

            if (self.data_mode == 0 or self.data_mode == 2):
                return output_cpu.to("cuda")
            else:
                return output_cpu

            # # # print("not in graph", x.shape[0])
            # output_cpu = torch.matmul(x.to("cpu"), weight.t())
            # if (bias != None):
            #     output_cpu += bias
            # outputs_gpu = output_cpu.to("cuda")
            #########################################################

# # Adepted from vllm/model_executor/layers/quantization/gptq.py: GPTQLinearMethod
class GPTQCpuLinearMethod(CpuLinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """
    FRAME_COUNT = 0

    def __init__(self, quant_config: GPTQConfig):
        # ��marlin����תΪ��ͨ������configҲ��Ҫת
        self.quant_config = self.convert_marlin2normal_config(quant_config)
        # offload�����Դ
        g_res = OffloadResource(1)
        self.cpu_launcher = g_res.get_launcher()
        self.cpu_linear = g_res.get_linear()
        self.cpu_memory_pool, self.cuda_memory_pool = g_res.get_memory_pool()

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
        print("GPTQCpuLinearMethod create_weights.\n")
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        output_size_per_partition = sum(output_partition_sizes)
        if (output_size_per_partition % self.quant_config.pack_factor.numerator
                != 0):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # For act-order models, we cannot use Exllama for row parallel layer
            if not self.quant_config.desc_act:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        # ��һά�����У��ڶ�ά�����С����������ǽ�input_size��kά�ȱ�����У����Լ���gemmʱ����Ҫ��ת���ˡ�
        # �������ά���ǵ�0ά������ǵ�1ά��8��һpack��ά���ǵ�0ά��
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor, # ��int32λ�棬pack_factor��8����һ��int32���8��int4.
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        # ������±꣺��input_sizeά�ȷ���(kά��)���飬ÿ��һ��group_sizeΪһ�顣
        g_idx = RowvLLMParameter(data=torch.tensor(
            [
                i // self.quant_config.group_size
                for i in range(input_size_per_partition)
            ],
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)
        qzeros_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }
        if scale_and_zero_input_dim is None:
            # ����в���
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            # �в��к��в��ж�����
            scales = GroupQuantScaleParameter(output_dim=1,
                                              input_dim=0,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        ####
        # self.input_cpu = torch.zeros(172, input_size_per_partition, dtype=torch.float16, device=torch.device('cpu'), pin_memory=True)
        # self.output_cpu = torch.zeros(172, output_size_per_partition, dtype=torch.float16, device=torch.device('cpu'), pin_memory=True)
        # self.output_gpu = torch.zeros(172, output_size_per_partition, dtype=torch.float16, device=torch.device('cuda'))
        
        self.inputs_cpu = []
        self.outputs_cpu = []
        self.outputs_gpu = []

        for m in range(1, 65):
            sub_tensor = self.cpu_memory_pool.allocate((m, input_size_per_partition), True)
            self.inputs_cpu.append(sub_tensor)
            sub_tensor = self.cpu_memory_pool.allocate((m, output_size_per_partition), False)
            self.outputs_cpu.append(sub_tensor)
            sub_tensor = self.cuda_memory_pool.allocate((m, output_size_per_partition), True)
            self.outputs_gpu.append(sub_tensor)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # for torch.compile
        GPTQCpuLinearMethod.FRAME_COUNT += 1
        print("GPTQCpuLinearMethod process_weights_after_loading: ", GPTQCpuLinearMethod.FRAME_COUNT)
        layer.qzeros = Parameter(layer.qzeros.data.to("cpu"), requires_grad=False)
        layer.qweight = Parameter(layer.qweight.data.to("cpu"), requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data.to("cpu"), requires_grad=False)
        layer.scales = Parameter(layer.scales.data.to("cpu"), requires_grad=False)

        # # ĳ�����ӣ�k=8960��n=1536
        # # Ȩ��һ��Ԫ�ش��ͬ������8�е�Ԫ�أ���qw[i,j]=w[i*8+(0~7), j]
        # # layer.qweight.shape torch.Size([1120, 1536])  1120 = 8960 // 8  int32��ԭά��[8960, 1536], ��ͨ���Բ�Ӧ����[1536, 8960]
        # # layer.scales.shape torch.Size([70, 1536])     70 = 8960 // 128 = 8960 / 8 / 16   fp16 ��ԭά��[70, 1536], ���鰴kά�Ȼ���
        # # layer.qzeros.shape torch.Size([70, 192])      192 = 1536 // 8��  int32��ԭά��[70, 1536]
        # # layer.g_idx.shape torch.Size([8960])
        # # in.shape: torch.Size([m, 8960])
        # # out.shape: torch.Size([m, 1536])
        #########
        # fp32
        self.weight_fp32_t = torch.zeros((layer.qweight.shape[1], layer.qweight.shape[0] * 8), dtype=torch.float32, device="cpu")
        self.cpu_launcher.submit(
            self.cpu_linear.UnpackW4ToFp32(
                layer.qweight.data_ptr(),
                layer.qweight.shape[0],
                layer.qweight.shape[1],
                layer.qzeros.data_ptr(),
                layer.scales.data_ptr(),
                self.weight_fp32_t.data_ptr()   # �����fp32��ͬʱ���ת�ã�ʹ���ֱ������linear���㡣
            )
        )
        

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # torch.cuda.is_current_stream_capturing()ΪTrue�����ֻ����capture�׶λ��replay�׶β������Ϊprint��cpu������
        # print("batch_size:{0}, stream:{1}, is_capture:{2}"
        #       .format(x.shape[0], torch.cuda.current_stream().cuda_stream, torch.cuda.is_current_stream_capturing()))
        idx = x.shape[0] - 1
        self.inputs_cpu[idx].copy_(x, non_blocking=True) # 

        self.cpu_launcher.submit_with_cuda_stream(
            torch.cuda.current_stream().cuda_stream,
            self.cpu_linear.forward(
                self.inputs_cpu[idx].shape[0],  # M
                self.outputs_cpu[idx].shape[1], # N
                self.inputs_cpu[idx].shape[1],  # K
                self.inputs_cpu[idx].data_ptr(), 
                self.weight_fp32_t.data_ptr(),
                0,
                self.outputs_cpu[idx].data_ptr()
            )
        )
        self.outputs_gpu[idx].copy_(self.outputs_cpu[idx], non_blocking=True) #
        if bias is not None:
            self.outputs_gpu[idx].add_(bias)
        return self.outputs_gpu[idx]

        # if not torch.cuda.is_current_stream_capturing():
        # print("out graph", x.shape[0])
        # output = F.linear(x.to("cpu").float(), self.weight_fp32_t).half().to('cuda')
        # # torch.set_printoptions(threshold=float('inf'), sci_mode=False)
        # # GPTQCpuLinearMethod.FRAME_COUNT += 1
        # # print("frame count ", GPTQCpuLinearMethod.FRAME_COUNT)
        # # print(output)
        # if bias is not None:
        #     output.add_(bias)
        # return output
        # output = (x.to("cpu") @ self.weight_fp16).to("cuda")

    def convert_marlin2normal_config(self, quant_config: GPTQMarlinConfig):
        weight_bits = 32 // quant_config.pack_factor
        group_size = quant_config.group_size
        desc_act = quant_config.desc_act
        lm_head_quantized = quant_config.lm_head_quantized
        return GPTQConfig(weight_bits, group_size, desc_act, lm_head_quantized)