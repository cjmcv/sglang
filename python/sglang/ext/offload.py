import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from typing import List
from sglang.ext.host_launcher import host_launcher


OFFLOAD2CPU = False
OFFLOAD2CPU_DATAMODE = 0 # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO

#######################
###      Usage      ###
# class UnquantizedLinearMethod(LinearMethodBase):
#     """Linear method without quantization."""

#     def create_weights(
#         self,
#         layer: torch.nn.Module,
#         input_size_per_partition: int,
#         output_partition_sizes: List[int],
#         input_size: int,
#         output_size: int,
#         params_dtype: torch.dtype,
#         **extra_weight_attrs,
#     ):
#         if (offload.OFFLOAD2CPU == False):
#             weight = Parameter(
#                 torch.empty(
#                     sum(output_partition_sizes),
#                     input_size_per_partition,
#                     dtype=params_dtype,
#                 ),
#                 requires_grad=False,
#             )
#         else:
#             self.offload_linear = offload.Linear()
#             weight = self.offload_linear.create_weights(input_size_per_partition, output_partition_sizes, params_dtype)
        
#         set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
#         layer.register_parameter("weight", weight)
#         set_weight_attrs(weight, extra_weight_attrs)

#     def apply(
#         self,
#         layer: torch.nn.Module,
#         x: torch.Tensor,
#         bias: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
        
#         if hasattr(self, 'offload_linear'):
#             return self.offload_linear.apply(x, layer.weight, bias)

#         return F.linear(x, layer.weight, bias)
class Linear():
    LINEAR_ON_CPU_CNT = 0
    HOST_LAUNCHER = host_launcher.HostLauncher()
    
    def create_weights(
        self,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        params_dtype: torch.dtype,
    ):           
        self.cpu_linear_idx = Linear.LINEAR_ON_CPU_CNT
        Linear.LINEAR_ON_CPU_CNT += 1 
        print("Create Linear", self.cpu_linear_idx, "on cpu")

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

        return weight

    def apply(self, x, weight, bias):
        # print("data_mode: ", self.data_mode)
        if torch.cuda.is_current_stream_capturing():
            # 0:GIGO, 1:GICO, 2:CIGO, 3:CICO 
            idx = x.shape[0] - 1
            print("apply cpu", idx, "stream", torch.cuda.current_stream().cuda_stream)
            self.inputs_cpu[idx].copy_(x, non_blocking=True)
            #######################################################
            M = self.inputs_cpu[idx].shape[0]
            bias_ptr = 0
            if (bias != None):
                bias_ptr = bias.data_ptr()
            Linear.HOST_LAUNCHER.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.linear.forward(
                    M,
                    self.inputs_cpu[idx].data_ptr(), 
                    weight.data_ptr(),
                    bias_ptr,
                    self.outputs_cpu[idx].data_ptr()
                )
            )
            #######################################################
            # print("output", self.outputs_cpu[idx])
            self.outputs_gpu[idx].copy_(self.outputs_cpu[idx], non_blocking=True)
            return self.outputs_gpu[idx]
        else:
            # # torch.set_printoptions(threshold=torch.inf)
            # idx = x.shape[0] - 1
            # self.inputs_cpu[idx].copy_(x)
            # M = self.inputs_cpu[idx].shape[0]
            # bias_ptr = 0
            # if (bias != None):
            #     bias_ptr = bias.data_ptr()
            # Linear.HOST_LAUNCHER.submit(
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
                output_cpu = F.linear(x.to("cpu"), weight, bias.to("cpu"))
            else:
                output_cpu = F.linear(x.to("cpu"), weight, None)
            outputs_gpu = output_cpu.to("cuda")

            return outputs_gpu
    