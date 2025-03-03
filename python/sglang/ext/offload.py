import torch
from torch.nn.parameter import Parameter

from typing import List
from sglang.ext.host_launcher import host_launcher


OFFLOAD2CPU = False


class Linear():
    HOST_LAUNCHER = host_launcher.HostLauncher()
    
    def create_weights(
        self,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        params_dtype: torch.dtype,
    ):            
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
        if torch.cuda.is_current_stream_capturing():
            idx = x.shape[0] - 1
            print("apply cpu", idx, "stream", torch.cuda.current_stream().cuda_stream)
            self.inputs_cpu[idx].copy_(x, non_blocking=True)
            #######################################################
            M = self.inputs_cpu[idx].shape[0]
            Linear.HOST_LAUNCHER.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                self.linear.forward(
                    M,
                    self.inputs_cpu[idx].data_ptr(), 
                    weight.data_ptr(),
                    self.outputs_cpu[idx].data_ptr()
                )
            )
            #######################################################
            # print("output", self.outputs_cpu[idx])
            self.outputs_gpu[idx].copy_(self.outputs_cpu[idx], non_blocking=True)
            # self.copy_event.record()
            # self.copy_event.wait()
            if (bias != None):
                self.outputs_gpu[idx] += bias
            return self.outputs_gpu[idx]
        else:
            # # torch.set_printoptions(threshold=torch.inf)
            # idx = x.shape[0] - 1
            # self.inputs_cpu[idx].copy_(x)
            # M = self.inputs_cpu[idx].shape[0]
            # Linear.HOST_LAUNCHER.submit(
            #     self.linear.forward(
            #         M,
            #         self.inputs_cpu[idx].data_ptr(), 
            #         weight.data_ptr(),
            #         self.outputs_cpu[idx].data_ptr()
            #     )
            # )
            # self.outputs_gpu[idx].copy_(self.outputs_cpu[idx])
            # if (bias != None):
            #     self.outputs_gpu[idx] += bias
            # return self.outputs_gpu[idx]
            ##########################################################

            print("not in graph", x.shape[0])
            output_cpu = torch.matmul(x.to("cpu"), weight.t())
            outputs_gpu = output_cpu.to("cuda")

            if (bias != None):
                outputs_gpu += bias
            return outputs_gpu
    