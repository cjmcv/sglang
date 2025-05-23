// Adapted from https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cuh
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace vllm {

// <NT> 36��block��ÿ��blockĬ��512���̡߳�
// 
constexpr int kMaxBlocks = 36;
// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;
// <NT> ����peer counter�����ֱ��Ӧ����ͬ������.
// ��Ϊ����ǰ GPU �߳̿���δͨ����һ��ͬ����ʱ���Ե� GPU �߳̿��п����ѵ���ڶ���ͬ���㡣
// ��ˣ�����ǰ GPU ��æ�ڵȴ�����������ʱ���Ե� GPU �����ѽ�����������Ϊ counter+1��
// ����ʹ�ý����������������������Ǳ�ڳ�ͻ��
struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  // Two sets of peer counters are needed for two syncs. The reason is that
  // it's possible for peer GPU block to arrive at the second sync point while
  // the current GPU block haven't passed the first sync point. Thus, peer GPU
  // may write counter+1 while current GPU is busy waiting for counter. We use
  // alternating counter array to avoid this possibility.
  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

// <NT> 8�ݣ���Ӧ����8��GPU
struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

// <NT> ͬ���źţ����8��gpu��ÿ��gpu������peer_counters
struct __align__(16) RankSignals {
  Signal* signals[8];
};

// <NT> alignof���ڲ�ѯ���ͻ�����Ķ���Ҫ��, ��alignof(float)Ϊ4��
// ����һ�� alignof(T) * sz �ֽڶ���ľ�̬����data��
// ��Ϊsz��ģ��������ڱ���׶ξͱ�ȷ��������T data[sz]�Ǿ�̬���顣
// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// <NT> ������alignof(T)*sz��
// ��P��alignof(T) * 16 / sizeof(T), ����16�ֽڶ��룬��128λ����������ld.128��st.128����gmem��smem��ȡ/д�롣
// ��A��alignof(float) * 16 / sizeof(T)�����T��2�ֽڣ���A��256λ�����T��4�ֽ�float������128λ��
// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

// <NT> ȷ����Ƶ����ǿ��inline����
#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) {
  return __half2float(val);
}

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) {
  return a += b;
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) {
  return __bfloat162float(val);
}
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

// <NT> һ��С����ļӷ�����
template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

// <NT> ��ȫ���ڴ�ִ�� ԭ���ͷŴ洢 ����������ͬ�����ƣ�st_flag_release��ld_flag_acquire��Ӧ��
// һ������������������ģʽ����д�̣߳���ɹؼ�����д���ʹ���ͷ��������ñ�־�����̣߳�ͨ����ȡ����ȴ���־��ȷ��������������
// st(�洢ָ��), release(ȷ��֮ǰ�������ڴ�������), sys(ϵͳ�ڴ�����), global(����ȫ���ڴ�)
// "r"��ͨ�üĴ�����32 λ�� 64 λ��ȡ���ڼܹ�����"r"(flag)��ʾflag ������ͨ�üĴ�����
// "l"��64 λ��ַ�Ĵ�����ר�����ڴ�Ѱַ����ARM GCC��"l" Ҳ���� 64 λֵ��"l"(flag_addr)��ʾflag_addr������ 64 λ��ֵַ
// "n"����������������offset��
static DINLINE void st_flag_release(FlagType* flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* flag_addr) {
  FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
  return flag;
}

// <NT> ��ͨ�Ĵ�ָ�ͨ��volatile��ֹ�������Ż������š�
static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// <NT> sg������GPU��ͬ���źţ�self_sg�ǵ�ǰGPU�Լ�����һ�ݣ���ʵҲ������sg�ֻ�ǵ�����ָ�˳�����
// threadIdx.x��Ӧһ��gpu��8��gpu��ֻ��0-7�������á�
// blockIdx.x����һά��������һ���߳̿鸺��һ��������ݿ����������block��0-7���̶߳�Ӧ�ļ�������ͬ�����ˣ���˵������gpu�ϸ�block��Ӧ���ݿ��ѿ�����ϡ�
// ͬ���ź�������ͨ����ĳ�Ա���������������Է�ֹ��ʹ�ø�ͨ����ѭ�������ͨ��ʱ����GPUͨ���ִβ�һ�£�������������
//   ��API���÷�ʽ��all_reduce(custom_ptr, inp1, out1, buffer_ptrs[rank], max_size)��buffer_ptrs��ע�ᣬѭ������ʱ�õ���ͬһ��buffer����ͬ������Ҳ�ᱻ��������buffer����
//   ��GPU_A����˵�һ��allreduce��������˵ڶ���ͨ�ţ���GPU_B���ڴ����һ�֣���Ϊ�������ͨ���õ���ͬһ��buffer��GPU_A�ڽ���ڶ���ʱû�еȴ�GPU_A�ĵ�һ�ֽ�����
// GPU_B�ڵڶ��ֿ�ͷ�ͻḲ��IPC buffer(sgl-kernel/csrc/allreduce/custom_all_reduce.cu -> all_reduce -> cudaMemcpyAsync)������GPU_A�ĵ�һ�������г�ͻ��
// ����barrier��blockΪ��λ������ʾҪִ�е�block�ﵽͬ��Ҫ�󼴿�ִ�и�blokc����һ��ͨ�ţ�����Ҫ�ȴ����嶼����˲ſ�ʼ���Ǹ�ϸ���ȵ�դ����
//   �����api�ĵ��÷�ʽ��all_reduce(custom_ptr, inp1, out1, None, max_size)��inp1��Ҫ����ע��ģ�ͬһ������ѭ������ʱ�õ���ͬһ��buffer�����ͬ�ϡ�������ǲ�ͬλ�õ�inp1��
// ���ڲ�ͬ��ַ������ּ������޹�����������Ҫͬ�������Ǻܶ��ֺ���������ͬһ��λ�õ���ͬ��ַ��inp1��barrierӦ���ͬһ��ַ����ͬ������Ӧ�÷ŵ�����api�ϣ�
// 
// ע��while �ڻ����ռ��GPU��Դ���������ʹ�ö���̻���߳̽���ģ�⣬whileѭ��ռ��GPU��Դ���ܻᵼ������һ���߳��޷���ɼ��㣬�����������ͬ����
// is_start: whether this is the very first synchronization barrier.
// need_fence: whether a memory fence is needed. If true, a release-acquire
// semantic is used to enforce memory access order before and after this
// barrier.
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  static_assert(!(is_start && need_fence));  // Start barrier shouldn't need fence.
  if (threadIdx.x < ngpus) {
    // Increment the counter. Technically we only need one counter, but we use
    // multiple per block to eliminate the need to share the counter via smem.
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
      st_flag_release(peer_counter_ptr, val);
      while (ld_flag_acquire(self_counter_ptr) != val)
        ;
    } else {
      st_flag_volatile(peer_counter_ptr, val);
      while (ld_flag_volatile(self_counter_ptr) != val)
        ;
    }
  }
  if constexpr (is_start || need_fence) __syncthreads();
}

// <NT> ptrs��������rank�Ĵ�ͨ�����ݵĶԵȹ����ڴ棬��һά�±���rank_id���ڶ�ά��ʵ��ͨ�����ݡ�
// P��ʵ��ͨ�ŵ��������ͣ�A��allreduce���Ӳ�����float���͡�
// upcast(ptrs[i][idx]) �Ὣ����ת����A��float���ͣ�Ȼ��ÿ��Ԫ����һ��128λ��С���飬ÿ��rank�Ķ�Ӧ�����ۼӣ�
// �ۼ���󽵻ص�ԭʼ���ͣ������output����Ϊoutput�Ǳ��ص�ַ��ÿ��rank�����������������õ���outputҲ����ȫһ�£�����Ҫ����ַ���
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

// <NT> __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)ָ���߳̿�ߴ����Դռ�õ�Լ����Ϣ��
// maxThreadsPerBlock�����������߳�����512��������allreduce����Ĭ�ϵ�Ҳ��512��
// minBlocksPerMultiprocessor��ÿ��SM������ͬʱפ������С�߳̿�������ȷ����������⣿
// multi_gpu_barrier��Ϊ�˶���ͨ���µĶ��GPU�Ľ�չͬ��������������ͨ�ţ���ǰGPU����˵�һ�ֻ����Ͽ�ʼ׼���ڶ��֣��ڵڶ��ּ���ǰ����������GPU���ڵ�һ��ͨ���У�����еȴ���
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_1stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

// <NT> RingAllReduce��ʵ�ְ汾��ScatterReduce + Allgather
// A[123456] B[abcdef] C[ijklmn] ���ݱ��ֳ�������: A[12,34,56] B[ab,cd,ef] C[ij,kl,mn]
// A��part��0�ţ�B����1�ţ�C��2��. ��allreduce��A�������0��part�õ�[12abij,34,56], B[ab,34cdkl,ef], C[ij,kl,56efmn]��ÿ��rank����һ��part���������ݣ�
// Ȼ��allgather��A��Ҫ��B�õ�1��part����C�õ�2��part������rank���ơ�
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // <NT> ��rank�������ȷֿ飬part��һ�����������start��ÿ�����ʼ�㡣
  // largest_part����17������8��gpu������7��partΪ2������1��partΪ3. largest_part = (17+2) % 8 = 3
  int part = size / ngpus;  
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  // <NT> rank3, ��Ӧtarget����34567012
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from all
  // ranks.
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IPC_KEY) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IPC_KEY) == alignof(cudaIpcMemHandle_t));

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  // <NT> ����rank��ͬ���ź�
  RankSignals sg_;
  // <NT> first���ⲿ������Դ��ַָ�룬second�Ƕ�Ӧ��������rank��peerָ�룬peerָ��ָ�����ڴ档
  // 
  // Stores an map from a pointer to its peer pointters from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  // <NT> ��ǰrank�����ͬ���ź�
  Signal* self_sg_;

  // Stores rank data from all ranks. This is mainly for cuda graph purposes.
  // For cuda graph to work, all kernel arguments must be fixed during graph
  // capture time. However, the peer pointers are not known during graph capture
  // time. Therefore, during capture, we increment the rank data pointer and use
  // that as the argument to the kernel. The kernel arguments are stored in
  // graph_unreg_buffers_. The actual peer pointers will be filled in at the
  // memory pointed to by the pointers in graph_unreg_buffers_ when
  // the IPC handles are exchanged between ranks.
  //
  // The overall process looks like this:
  // 1. Graph capture.
  // 2. Each rank obtains the IPC handles for each addresses used during cuda
  // graph capture using get_graph_buffer_ipc_meta.
  // 3. (In Python) all gather the IPC handles.
  // 4. Obtain the peer pointers by opening the IPC handles, and store them in
  // the rank data array at corresponding positions.
  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  // a map from IPC handles to opened IPC pointers
  std::map<IPC_KEY, char*> ipc_handles_;

  /**
   * Signals are an array of ipc-enabled buffers from all ranks.
   * For each of the buffer, the layout is as follows:
   * | -- sizeof(Signal) -- | ------ a few MB ----- |
   * The first section is for allreduce synchronization, and the second section
   * is for storing the intermediate results required by some allreduce algos.
   *
   * Note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor.
   */
  CustomAllreduce(
      Signal** signals, void* rank_data, size_t rank_data_sz, int rank, int world_size, bool full_nvlink = true)
      : rank_(rank),
        world_size_(world_size),
        full_nvlink_(full_nvlink),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
    }
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(
          (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle), cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  // <NT> ��captureʱ����allreduce�ӿڵ������tensor��ָ�붼������graph_unreg_buffers_���ڽ���capture����øú�����ͨ��cudaIpcGetMemHandle��������Ϊ�����ڴ档
  //  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR ���Եõ� ptr ָ����ڴ����׵�ַ ��
  //  (�ڴ���Ҫ��cudaMallocManaged�����ͳһ�����ַUVA �� pytorch Tensor������Դ�(pytorch�з�װ�����Ϣ)�� ��������cudaMalloc�����)��
  //  ������cudaMallocManaged������һ����ڴ棬�� base_ptr ָ����Ȼ����ڵ�ַƫ�ƽ��ڴ滮�ֳɶ�飬�䵱���tensor��
  //  ��ÿ������׵�ַͨ��CU_POINTER_ATTRIBUTE_RANGE_START_ADDR�����������õ�ƫ��ǰ�� base_ptr��
  //
  // handles��Ż����׵�ַ���ڴ湲��õ��ľ�������ǹ����ڴ���׵�ַ��ƫ�Ƶ�ַ - �׵�ַ = offset�������ڴ���׵�ַ����ƫ���������õ�ÿ��tensor�ڹ����ڴ��ϵĵ�ַ��
  // ��Ϊ��������tensor������ͬһ����ڴ棬����graph_unreg_buffers_���׵�ַҲ���кܶ��������Ԫ����ͬһ���ڴ治ͬoffset������Ԫ���ǲ�ͬ�ڴ�顣
  // 
  // ������·��python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
  //     capture -> (final) register_graph_buffers -> ops.get_graph_buffer_ipc_meta(��ǰ����) -> ops.register_graph_buffers
  //     ����capture��������ã�capture�л�ͨ������ allreduce �� status == cudaStreamCaptureStatusActive����capture�������tensorȫ�����뵽graph_unreg_buffers_�С�
  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      // note: must share the base address of each allocation, or we get wrong
      // address
      if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error(
          "Rank data buffer is overflowed by " + std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
  }

  /**
   * <NT> d_rank_data_base_ �ǹ���ʱ�������ͨ�Դ��ַ��ר�����ڵ�ǰrank����ָͨ�뵽����rank�Ĺ���ָ���ӳ���ϵ��
   * (Ҳ�ο�python/sglang/srt/distributed/device_communicators/custom_all_reduce.py��create_shared_buffer�� ��ǰrank���������ָͨ�룬
   * ����rank����Ҫ��cudaIpcGetMemHandle/cudaIpcOpenMemHandle���⴦��)
   * ������� ptrs ���� custom_all_reduce.py��buffer_ptrs��������Ҫͨ�ŵ�Ŀ�����ݵĵ�ַ��һ��С�����������rank�Ĺ����ڴ��ַ������ʹ��rank_data�������Щָ�����顣
   * ��Ϊ���ӳ���ϵֻ��Ҫ���ؿɼ���������torch.empty����ռ伴�ɡ�
   * 
   * buffers_��first��ͨ���õ�Ŀ�����ݵĵ�ǰrank���ڴ��ַ��second��Ŀ��ͨ�����ݵ�����rank�Ĺ����ڴ��ַ(�����������)��
   * ʵ��ʹ��ʱ�����뵱ǰrank���ڴ��ַ���Ϳ���ͨ��buffers_ȡ�����Ӧ������rank�ĶԵȵ��ڴ��ַ��ֱ�ӻ�����Щ�ڴ���й�Լ�������ɡ�
   * 
   * Register already-shared IPC pointers.
   */
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) {
      data.ptrs[i] = ptrs[i];
    }
    auto d_data = d_rank_data_base_++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  // Note: when registering graph buffers, we intentionally choose to not
  // deduplicate the addresses. That means if the allocator reuses some
  // addresses, they will be registered again. This is to account for the remote
  // possibility of different allocation patterns between ranks. For example,
  // rank 1 may get the same input address for the second allreduce, but rank 2
  // got a different address. IPC handles have internal reference counting
  // mechanism so overhead should be small.
  void
  register_graph_buffers(const std::vector<std::string>& handles, const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (int i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle = open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    // <NT> �ʣ�d_rank_data_base_��cuda graph��replay�׶Σ����ʹ�ã�
    // ����Ϊ����allreduce������������capture���Ѿ����̻������Բ�������ʽ�����С���replayʱ���ϸ���capture�е����̽������С�
    // graph_unreg_buffers_������Ҳ�Ѿ������񵽣�������replayʱ����Ȼ���Կ�������if (status == cudaStreamCaptureStatusActive)�����ݣ�
    // ���Ե����ض��ĵ���λ�ã������ض���graph_unreg_buffers_.size()��ͨ��ptrs = d_rank_data_base_ + graph_unreg_buffers_.size()��
    // �ܹ�׼ȷ�õ��ض�����λ�õ�����tensor��Ӧ�ĵ�ַ��
    CHECK_CUDA_SUCCESS(
        cudaMemcpy(d_rank_data_base_, rank_data.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  /**
   * Performs allreduce, assuming input has already been registered.
   *
   * Block and grid default configs are results after careful grid search. Using
   * 36 blocks give the best or close to the best runtime on the devices I
   * tried: A100, A10, A30, T4, V100. You'll notice that NCCL kernels also only
   * take a small amount of SMs. Not quite sure the underlying reason, but my
   * guess is that too many SMs will cause contention on NVLink bus.
   */
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size, int threads = 512, int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "custom allreduce currently requires input length to be multiple "
          "of " +
          std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error(
          "max supported block limit is " + std::to_string(kMaxBlocks) + ". Got " + std::to_string(block_limit));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));
    // <NT> ������cuda graph����capture��ʱ�����ⲿ�ṩ������tensor���뵽graph_unreg_buffers_�С�
    // rank_data ��ŵ���Ŀ������ �� ��ǰrank���Դ�ָ�� ӳ�䵽 ����rank�ĵ�ַ���� ��ӳ���ϵ, ���cuda graphֱ����graph_unreg_buffers_.size()�䵱ƫ������
    if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) + " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    // <NT> 0�ǹ����ڴ��С��ָ��kernel����Ҫ��̬���乲���ڴ棬�����ų����澲̬ʹ�ù����ڴ档
#define KL(ngpus, name) name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
    // TODO(hanzhi713): Threshold is different for A100 and H100.
    // Add per device threshold.
#define REDUCE_CASE(ngpus)                                                                        \
  case ngpus: {                                                                                   \
    if (world_size_ == 2) {                                                                       \
      KL(ngpus, cross_device_reduce_1stage);                                                      \
    } else if (full_nvlink_) {                                                                    \
      if ((world_size_ <= 4 && bytes < 512 * 1024) || (world_size_ <= 8 && bytes < 256 * 1024)) { \
        KL(ngpus, cross_device_reduce_1stage);                                                    \
      } else {                                                                                    \
        KL(ngpus, cross_device_reduce_2stage);                                                    \
      }                                                                                           \
    }                                                                                             \
    break;                                                                                        \
  }

    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
            "gpus = " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~CustomAllreduce() {
    for (auto [_, ptr] : ipc_handles_) {
      CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
    }
  }
};
/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and add
 a template instantiation:
 * template void vllm::CustomAllreduce::allreduce<half>(cudaStream_t, half *,
 half *, int, int, int);
*/
}  // namespace vllm
