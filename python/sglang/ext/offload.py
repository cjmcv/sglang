from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

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
# from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
# #

OFFLOAD2CPU = False
OFFLOAD2CPU_DATAMODE = 0 # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO

# 对应from sglang.srt.layers.linear import LinearMethodBase
# 新写一份防止import成环
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
    
# 对标UnquantizedLinearMethod
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
        # TODO: 为应对cudagraph的各个bs，开太多内存，考虑减少减低内存开销。
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
            

# class AWQCpuLinearMethod(CpuLinearMethodBase):
#     """Linear method for AWQ Marlin.

#     Args:
#         quant_config: The AWQ Marlin quantization config.
#     """

#     def __init__(self, quant_config: AWQMarlinConfig) -> None:
#         self.quant_config = quant_config

#     def create_weights(
#         self,
#         layer: torch.nn.Module,
#         input_size_per_partition: int,
#         output_partition_sizes: List[int],
#         input_size: int,
#         output_size: int,
#         params_dtype: torch.dtype,
#         **extra_weight_attrs,
#     ) -> None:
#         del output_size
#         output_size_per_partition = sum(output_partition_sizes)
#         weight_loader = extra_weight_attrs.get("weight_loader")

#         # Normalize group_size
#         if self.quant_config.group_size != -1:
#             group_size = self.quant_config.group_size
#         else:
#             group_size = input_size

#         verify_marlin_supports_shape(
#             output_size_per_partition=output_size_per_partition,
#             input_size_per_partition=input_size_per_partition,
#             input_size=input_size,
#             group_size=group_size)

#         qweight = PackedvLLMParameter(
#             data=torch.empty(
#                 input_size_per_partition,
#                 output_size_per_partition // self.quant_config.pack_factor,
#                 dtype=torch.int32,
#             ),
#             input_dim=0,
#             output_dim=1,
#             packed_dim=1,
#             packed_factor=self.quant_config.pack_factor,
#             weight_loader=weight_loader)

#         num_groups = input_size_per_partition // group_size

#         qzeros = PackedvLLMParameter(
#             data=torch.empty(
#                 num_groups,
#                 output_size_per_partition // self.quant_config.pack_factor,
#                 dtype=torch.int32,
#             ),
#             input_dim=0,
#             output_dim=1,
#             packed_dim=1,
#             packed_factor=self.quant_config.pack_factor,
#             weight_loader=weight_loader)

#         scales = GroupQuantScaleParameter(data=torch.empty(
#             num_groups,
#             output_size_per_partition,
#             dtype=params_dtype,
#         ),
#                                           input_dim=0,
#                                           output_dim=1,
#                                           weight_loader=weight_loader)

#         layer.register_parameter("qweight", qweight)
#         layer.register_parameter("qzeros", qzeros)
#         layer.register_parameter("scales", scales)

#         layer.input_size_per_partition = input_size_per_partition
#         layer.output_size_per_partition = output_size_per_partition
#         layer.num_groups = num_groups

#     # TODO: Update this docs
#     # Checkpoints are serialized in AutoAWQ format, which is different from the
#     # marlin format. This function is called after the weights are loaded.
#     # Here, we handle the repacking
#     def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
#         device = layer.qweight.device
#         layer.qweight = torch.nn.Parameter(layer.qweight.data,
#                                            requires_grad=False)
#         layer.qzeros = torch.nn.Parameter(layer.qzeros.data,
#                                           requires_grad=False)
#         layer.scales = torch.nn.Parameter(layer.scales.data,
#                                           requires_grad=False)

#         # Allocate marlin workspace
#         layer.workspace = marlin_make_workspace(
#             layer.output_size_per_partition, device)

#         # Repack weights from AWQ format to marlin format.
#         marlin_qweight = ops.awq_marlin_repack(
#             layer.qweight,
#             size_k=layer.input_size_per_partition,
#             size_n=layer.output_size_per_partition,
#             num_bits=self.quant_config.quant_type.size_bits)
#         replace_parameter(layer, "qweight", marlin_qweight)

#         # Permute scales from AWQ format to marlin format.
#         marlin_scales = marlin_permute_scales(
#             layer.scales,
#             size_k=layer.input_size_per_partition,
#             size_n=layer.output_size_per_partition,
#             group_size=self.quant_config.group_size)
#         replace_parameter(layer, "scales", marlin_scales)

#         # Permute zero-points from AWQ format to marlin format.
#         marlin_zp = awq_to_marlin_zero_points(
#             layer.qzeros,
#             size_k=layer.num_groups,
#             size_n=layer.output_size_per_partition,
#             num_bits=self.quant_config.quant_type.size_bits)
#         replace_parameter(layer, "qzeros", marlin_zp)

#         # Not-used
#         layer.g_idx = marlin_make_empty_g_idx(device)
#         layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

#     def apply(
#         self,
#         layer: torch.nn.Module,
#         x: torch.Tensor,
#         bias: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         return apply_awq_marlin_linear(
#             input=x,
#             weight=layer.qweight,
#             weight_scale=layer.scales,
#             weight_zp=layer.qzeros,
#             g_idx=layer.g_idx,
#             g_idx_sort_indices=layer.g_idx_sort_indices,
#             workspace=layer.workspace,
#             quant_type=self.quant_config.quant_type,
#             output_size_per_partition=layer.output_size_per_partition,
#             input_size_per_partition=layer.input_size_per_partition,
#             bias=bias)