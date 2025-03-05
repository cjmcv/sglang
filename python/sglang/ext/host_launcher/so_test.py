import host_launcher
import torch
import numpy as np

launcher = host_launcher.HostLauncher()
launcher.hello(2)


M = 6
in_features = 1536     # K
out_features = 17920   # N
config = host_launcher.linear.LinearConfig(in_features, out_features)
linear = host_launcher.linear.Linear(config)


input_cpu = torch.rand(M, in_features, dtype=torch.bfloat16, device=torch.device('cpu'), pin_memory=True)
weight = torch.rand(1, out_features, in_features, dtype=torch.bfloat16, device=torch.device('cpu'), pin_memory=True)
output_cpu = torch.ones(M, out_features, dtype=torch.bfloat16, device=torch.device('cpu'), pin_memory=True)

input_cpu = 2 * input_cpu - 1
weight = 2 * weight - 1

torch.set_printoptions(threshold=float('inf'))
linear.forward_pure(
    M,
    input_cpu.data_ptr(), 
    weight.data_ptr(),
    output_cpu.data_ptr(),
    0
)
print(output_cpu)
result0 = output_cpu

linear.forward_pure(
    M,
    input_cpu.data_ptr(), 
    weight.data_ptr(),
    output_cpu.data_ptr(),
    1
)
print(output_cpu)
result1 = output_cpu

diff = result1 - result0
print("diff", diff)