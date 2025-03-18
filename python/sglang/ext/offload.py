from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Optional

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sglang.srt.utils import set_weight_attrs
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from sglang.ext.host_launcher import host_launcher

# # AWQ
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
##
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig, choose_mp_linear_kernel)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported, marlin_moe_permute_scales,
    marlin_repeat_scales_on_all_ranks, verify_marlin_supported)
##
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.model_executor.layers.quantization.gptq import GPTQConfig, ExllamaState
from vllm import _custom_ops as ops
# #

OFFLOAD2CPU = False
OFFLOAD2CPU_DATAMODE = 0 # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO

# gptq���ɿӵ㣺1. gptq��zero��Ҫ����+1���ο�autogptq�Ĵ���ó���
#              2. gptq��Ȩ���Ƿ�ת�õģ�����Ҫ��linear����㣬��Ҫת�á���ִ��weight_t = weight.t(), Ȼ��ʹ��weight_t.data_ptr()���뵽cpp�У��ᷢ�ֵõ������ݲ�û��ת�ã�
#          

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
    HOST_LAUNCHER = host_launcher.HostLauncher()
    
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
        # TODO: ΪӦ��cudagraph�ĸ���bs����̫���ڴ棬���Ǽ��ټ����ڴ濪����
        self.input_cpu = torch.zeros(172, input_size_per_partition, dtype=params_dtype, device=torch.device('cpu'), pin_memory=True)
        self.output_cpu = torch.zeros(172, sum(output_partition_sizes), dtype=params_dtype, device=torch.device('cpu'), pin_memory=True)
        self.output_gpu = torch.zeros(172, sum(output_partition_sizes), dtype=params_dtype, device=torch.device('cuda'))
        
        self.inputs_cpu = []
        self.outputs_cpu = []
        self.outputs_gpu = []

        for shape in range(1, 172):
            sub_tensor = self.input_cpu[:shape]
            self.inputs_cpu.append(sub_tensor)
            sub_tensor = self.output_cpu[:shape]
            self.outputs_cpu.append(sub_tensor)
            sub_tensor = self.output_gpu[:shape]
            self.outputs_gpu.append(sub_tensor)

        self.copy_event = torch.cuda.Event(enable_timing=False)
        config = host_launcher.linear.LinearConfig(input_size_per_partition, sum(output_partition_sizes)) # in_features, out_features
        self.linear = host_launcher.linear.Linear(config)

        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("data_mode: ", self.data_mode)
        if torch.cuda.is_current_stream_capturing():
            # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO 
            idx = x.shape[0] - 1
            print("apply cpu", idx, "stream", torch.cuda.current_stream().cuda_stream)
            if (self.data_mode == 0 or self.data_mode == 1):
                self.inputs_cpu[idx].copy_(x, non_blocking=True)
                input = self.inputs_cpu[idx]
            else:
                input = x
            #######################################################
            bias_ptr = 0
            if (bias != None):
                bias_ptr = bias.data_ptr()
            UnquantizedCpuLinearMethod.HOST_LAUNCHER.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.linear.forward(
                    input.shape[0],
                    input.data_ptr(), 
                    layer.weight.data_ptr(),
                    bias_ptr,
                    self.outputs_cpu[idx].data_ptr()
                )
            )
            #######################################################
            # print("output", self.outputs_cpu[idx])
            if (self.data_mode == 0 or self.data_mode == 2):
                self.outputs_gpu[idx].copy_(self.outputs_cpu[idx], non_blocking=True)
                return self.outputs_gpu[idx]
            else:
                return self.outputs_cpu[idx]
        else:
            # # torch.set_printoptions(threshold=torch.inf)
            # idx = x.shape[0] - 1
            # self.inputs_cpu[idx].copy_(x)
            # M = self.inputs_cpu[idx].shape[0]
            # bias_ptr = 0
            # if (bias != None):
            #     bias_ptr = bias.data_ptr()
            # UnquantizedCpuLinearMethod.HOST_LAUNCHER.submit(
            #     self.linear.forward(
            #         M,
            #         self.inputs_cpu[idx].data_ptr(), 
            #         weight.data_ptr(),
            #         bias_ptr,
            #         self.outputs_cpu[idx].data_ptr()
            #     )
            # )
            # self.outputs_gpu[idx].copy_(self.outputs_cpu[idx])
            # return self.outputs_gpu[idx]
            #########################################################

            # # # print("not in graph", x.shape[0])
            # output_cpu = torch.matmul(x.to("cpu"), weight.t())
            # if (bias != None):
            #     output_cpu += bias
            # outputs_gpu = output_cpu.to("cuda")
            #########################################################
            if (bias != None):
                output_cpu = F.linear(x.to("cpu"), layer.weight, bias.to("cpu"))
            else:
                output_cpu = F.linear(x.to("cpu"), layer.weight, None)

            if (self.data_mode == 0 or self.data_mode == 2):
                return output_cpu.to("cuda")
            else:
                return output_cpu

# # Adepted from vllm/model_executor/layers/quantization/gptq.py: GPTQLinearMethod
class GPTQCpuLinearMethod(CpuLinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """
    FRAME_COUNT = 0
    HOST_LAUNCHER = host_launcher.HostLauncher()
    config = host_launcher.linear.LinearConfig(1) 
    cpu_linear = host_launcher.linear.Linear(config) # TODO: �䵱��̬ʵ��ʱ���������ʱ�ɿ�ס(�ȴ���)

    def __init__(self, quant_config: GPTQConfig):
        # ��marlin����תΪ��ͨ������configҲ��Ҫת
        self.quant_config = self.convert_marlin2normal_config(quant_config)

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
        exllama_state = ExllamaState.UNINITIALIZED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (input_size != input_size_per_partition
                and self.quant_config.group_size != -1):
            # For act-order models, we cannot use Exllama for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
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

        # print(".scales.device", scales.device)
        # print(".qzeros.device", qzeros.device)
        # print(".qweight.device", qweight.device)
        # print(".g_idx.device", g_idx.device)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.exllama_state = exllama_state

        ####
        self.input_cpu = torch.zeros(172, input_size_per_partition, dtype=torch.float16, device=torch.device('cpu'), pin_memory=True)
        self.output_cpu = torch.zeros(172, output_size_per_partition, dtype=torch.float16, device=torch.device('cpu'), pin_memory=True)
        self.output_gpu = torch.zeros(172, output_size_per_partition, dtype=torch.float16, device=torch.device('cuda'))
        
        self.inputs_cpu = []
        self.outputs_cpu = []
        self.outputs_gpu = []

        for shape in range(1, 172):
            sub_tensor = self.input_cpu[:shape]
            self.inputs_cpu.append(sub_tensor)
            sub_tensor = self.output_cpu[:shape]
            self.outputs_cpu.append(sub_tensor)
            sub_tensor = self.output_gpu[:shape]
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

        # #########
        # # fp16
        # self.weight_fp16 = torch.zeros((layer.qweight.shape[0] * 8, layer.qweight.shape[1]), dtype=torch.float16, device="cpu")
        # GPTQCpuLinearMethod.HOST_LAUNCHER.submit(
        #     self.linear.UnpackW4ToFp16(
        #         layer.qweight.data_ptr(),
        #         layer.qweight.shape[0],
        #         layer.qweight.shape[1],
        #         layer.qzeros.data_ptr(),
        #         layer.scales.data_ptr(),
        #         self.weight_fp16.data_ptr()
        #     )
        # )
        #########
        # fp32
        self.weight_fp32_t = torch.zeros((layer.qweight.shape[1], layer.qweight.shape[0] * 8), dtype=torch.float32, device="cpu")
        GPTQCpuLinearMethod.HOST_LAUNCHER.submit(
            GPTQCpuLinearMethod.cpu_linear.UnpackW4ToFp32(
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
        
        # output = F.linear(x.to("cpu").float(), self.weight_fp32).half().to('cuda')
        # torch.set_printoptions(threshold=float('inf'), sci_mode=False)
        # GPTQCpuLinearMethod.FRAME_COUNT += 1
        # print("frame count ", GPTQCpuLinearMethod.FRAME_COUNT)
        # print(output)
        # if bias is not None:
        #     output.add_(bias)
        # return output
    
        idx = x.shape[0] - 1
        self.inputs_cpu[idx].copy_(x, non_blocking=True) # 

        UnquantizedCpuLinearMethod.HOST_LAUNCHER.submit_with_cuda_stream(
            torch.cuda.current_stream().cuda_stream,
            GPTQCpuLinearMethod.cpu_linear.forward(
                self.inputs_cpu[idx].shape[0],  # M
                self.outputs_cpu[idx].shape[1], # N
                self.inputs_cpu[idx].shape[1],  # K
                self.inputs_cpu[idx].data_ptr(), 
                self.weight_fp32_t.data_ptr(),
                0,
                self.outputs_cpu[idx].data_ptr()
            )
        )
        
        # torch.set_printoptions(threshold=float('inf'), sci_mode=False)
        # GPTQCpuLinearMethod.FRAME_COUNT += 1
        # print("frame count ", GPTQCpuLinearMethod.FRAME_COUNT)
        # # print("before", self.weight_fp32)
        # # print("after", self.weight_fp32_t)
        # print(self.outputs_cpu[idx])
        self.outputs_gpu[idx].copy_(self.outputs_cpu[idx], non_blocking=True) #
        if bias is not None:
            self.outputs_gpu[idx].add_(bias)
        return self.outputs_gpu[idx]
    
        
        # output = (x.to("cpu") @ self.weight_fp16).to("cuda")
        # torch.set_printoptions(threshold=float('inf'), sci_mode=False)
        # GPTQCpuLinearMethod.FRAME_COUNT += 1
        # print("frame count ", GPTQCpuLinearMethod.FRAME_COUNT)
        # print(output)

        # if bias is not None:
        #     output.add_(bias)
        # return output

    def convert_marlin2normal_config(self, quant_config: GPTQMarlinConfig):
        weight_bits = 32 // quant_config.pack_factor
        group_size = quant_config.group_size
        desc_act = quant_config.desc_act
        lm_head_quantized = quant_config.lm_head_quantized
        return GPTQConfig(weight_bits, group_size, desc_act, lm_head_quantized)