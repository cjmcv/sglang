// Adapted from: https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cu
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "custom_all_reduce.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

// <NT> allreduce 算法的初始化函数，ar是allreduce的缩写。
// fake_ipc_ptrs是python端传入的meta，包含所有rank的共享内存(自身rank的是cudaMalloc申请的，其他rank是cudaIpcGetMemHandle)，
//              每份大小是 sizeof(Signal) + max_size, sizeof(Signal)存放的是同步用的Signal， max_size部分存放的是allreduce算法的一些中间结果。
// rank_data是本地使用torch.empty申请的显存空间，存放的是目标数据 从 当前rank的显存指针 映射到 所有rank的地址数组 的映射关系。
//          当一份数据需要通信，要先调用到create_shared_buffer为其申请显存空间，同时获取得到其他rank的对等共享内存地址，然后注册到 buffers_ 里。
//          实际通信时，接口传入当前rank的内存地址，通过buffers_得到所有rank的共享内存地址，然后在这些地址上进行数据操作。
fptr_t
init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank, bool full_nvlink) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");

  sglang::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<sglang::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new sglang::CustomAllreduce(
      ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size, full_nvlink);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float: {
      fa->allreduce<float>(
          stream, reinterpret_cast<float*>(reg_buffer), reinterpret_cast<float*>(out.data_ptr()), out.numel());
      break;
    }
    case at::ScalarType::Half: {
      fa->allreduce<half>(
          stream, reinterpret_cast<half*>(reg_buffer), reinterpret_cast<half*>(out.data_ptr()), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream,
          reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()),
          out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<sglang::CustomAllreduce*>(_fa);
}

int64_t meta_size() {
  return sizeof(sglang::Signal);
}

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}
