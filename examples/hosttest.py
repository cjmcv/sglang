import torch
import cuda.bindings.runtime as cbr
import numpy as np
import ctypes

class cb_gemm_param(ctypes.Structure):
    _fields_ = [
        ("a", ctypes.c_void_p),
        ("b", ctypes.c_void_p),
        ("c", ctypes.c_void_p),
        ("m", ctypes.c_int),
        ("n", ctypes.c_int),
        ("k", ctypes.c_int),
    ]

def host_gemm_callback(user_data):
    data = cb_gemm_param.from_address(user_data)
    print(data.m, data.n, data.k)

    element_size = 2 # for bfloat16
    a_buffer = (ctypes.c_byte * element_size * data.m * data.k).from_address(data.a)
    a = torch.frombuffer(a_buffer, dtype=torch.bfloat16).reshape(data.m, data.k)

    b_buffer = (ctypes.c_byte * element_size * data.n * data.k).from_address(data.b)
    b = torch.frombuffer(b_buffer, dtype=torch.bfloat16).reshape(data.n, data.k)
    
    c_buffer = (ctypes.c_byte * element_size * data.m * data.n).from_address(data.c)
    c = torch.frombuffer(c_buffer, dtype=torch.bfloat16).reshape(data.m, data.n)

    torch.matmul(a, b.t(), out=c)
    print(c)
    print(a.shape, b.shape, c.shape)
    return 0

CALLBACK_TYPE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
c_callback = CALLBACK_TYPE(host_gemm_callback)
callback = cbr.cudaHostFn_t(_ptr=ctypes.addressof(c_callback))

stream = torch.cuda.Stream()
# err, stream = cbr.cudaStreamCreate()

m = 128
n = 256
k = 512
a = torch.randn(m, k, dtype=torch.bfloat16)
b = torch.randn(k, n, dtype=torch.bfloat16)
c = torch.zeros(m, n, dtype=torch.bfloat16)
usr_data = cb_gemm_param(a.data_ptr(), b.data_ptr(), c.data_ptr(), m, n, k)

print("element_size", a.element_size())

err, = cbr.cudaLaunchHostFunc(stream.cuda_stream, callback, ctypes.addressof(usr_data))

err, = cbr.cudaStreamSynchronize(stream.cuda_stream)

print("Matrix multiplication result:")
print(c)

err, = cbr.cudaStreamDestroy(stream.cuda_stream)