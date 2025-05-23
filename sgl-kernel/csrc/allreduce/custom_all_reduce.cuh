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

// <NT> 36个block，每个block默认512个线程。
// 
constexpr int kMaxBlocks = 36;
// Counter may overflow, but it's fine since unsigned int overflow is
// well-defined behavior.
using FlagType = uint32_t;
// <NT> 两份peer counter集，分别对应两次同步操作.
// 因为当当前 GPU 线程块尚未通过第一个同步点时，对等 GPU 线程块有可能已到达第二个同步点。
// 因此，当当前 GPU 正忙于等待计数器更新时，对等 GPU 可能已将计数器递增为 counter+1。
// 所以使用交替计数器数组来消除这种潜在冲突。
struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  // Two sets of peer counters are needed for two syncs. The reason is that
  // it's possible for peer GPU block to arrive at the second sync point while
  // the current GPU block haven't passed the first sync point. Thus, peer GPU
  // may write counter+1 while current GPU is busy waiting for counter. We use
  // alternating counter array to avoid this possibility.
  alignas(128) FlagType peer_counter[2][kMaxBlocks][8];
};

// <NT> 8份，对应最多的8个GPU
struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

// <NT> 同步信号，最多8个gpu，每个gpu有两份peer_counters
struct __align__(16) RankSignals {
  Signal* signals[8];
};

// <NT> alignof用于查询类型或变量的对齐要求, 如alignof(float)为4。
// 构建一个 alignof(T) * sz 字节对齐的静态数组data，
// 因为sz是模板参数，在编译阶段就被确定，所以T data[sz]是静态数组。
// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// <NT> 对齐是alignof(T)*sz，
// 则P是alignof(T) * 16 / sizeof(T), 就是16字节对齐，共128位，方便生成ld.128和st.128，从gmem或smem读取/写入。
// 而A是alignof(float) * 16 / sizeof(T)，如果T是2字节，则A是256位，如果T是4字节float，则是128位。
// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

// <NT> 确保高频函数强制inline调用
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

// <NT> 一份小数组的加法操作
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

// <NT> 在全局内存执行 原子释放存储 操作，用于同步机制，st_flag_release和ld_flag_acquire对应。
// 一般用于生产者消费者模式，如写线程：完成关键数据写入后，使用释放语义设置标志。读线程：通过获取语义等待标志，确保看到最新数据
// st(存储指令), release(确保之前的所有内存操作完成), sys(系统内存屏障), global(操作全局内存)
// "r"：通用寄存器（32 位或 64 位，取决于架构），"r"(flag)表示flag 必须在通用寄存器中
// "l"：64 位地址寄存器（专用于内存寻址），ARM GCC的"l" 也用于 64 位值，"l"(flag_addr)表示flag_addr必须是 64 位地址值
// "n"：立即数常量（如offset）
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

// <NT> 普通的存指令，通过volatile禁止编译器优化和重排。
static DINLINE void st_flag_volatile(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_volatile(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// <NT> sg是所有GPU的同步信号，self_sg是当前GPU自己的那一份，其实也包含在sg里，只是单独被指了出来。
// threadIdx.x对应一个gpu，8个gpu则只有0-7号起作用。
// blockIdx.x做第一维度索引，一个线程块负责一个块的数据拷贝，如果该block的0-7号线程对应的计数器都同步上了，则说明各个gpu上该block对应数据块已拷贝完毕。
// 同步信号是整个通信类的成员变量，这样做可以防止，使用该通信类循环做多次通信时，因GPU通信轮次不一致，导致数据有误。
//   按API调用方式：all_reduce(custom_ptr, inp1, out1, buffer_ptrs[rank], max_size)，buffer_ptrs已注册，循环调用时用的是同一个buffer（不同的输入也会被拷贝到该buffer）。
//   如GPU_A完成了第一轮allreduce，想进入了第二轮通信，而GPU_B还在处理第一轮，因为如果两轮通信用的是同一份buffer，GPU_A在进入第二轮时没有等待GPU_A的第一轮结束，
// GPU_B在第二轮开头就会覆盖IPC buffer(sgl-kernel/csrc/allreduce/custom_all_reduce.cu -> all_reduce -> cudaMemcpyAsync)，导致GPU_A的第一轮数据有冲突。
// 而且barrier以block为单位，即表示要执行的block达到同步要求即可执行该blokc的下一轮通信，不需要等待整体都完成了才开始，是个细粒度的栅栏。
//   如果按api的调用方式：all_reduce(custom_ptr, inp1, out1, None, max_size)，inp1需要是已注册的，同一份输入循环调用时用的是同一个buffer，情况同上。而如果是不同位置的inp1，
// 属于不同地址，则多轮计算则无关联，不再需要同步？除非很多轮后又碰到了同一个位置的相同地址的inp1。barrier应针对同一地址进行同步，不应该放到整体api上？
// 
// 注意while 内会持续占用GPU资源，如果单卡使用多进程或多线程进行模拟，while循环占用GPU资源可能会导致另外一个线程无法完成计算，导致难以完成同步。
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

// <NT> ptrs包含所有rank的待通信数据的对等共享内存，第一维下标是rank_id，第二维是实际通信数据。
// P是实际通信的数据类型，A是allreduce叠加操作的float类型。
// upcast(ptrs[i][idx]) 会将数据转换成A的float类型，然后每个元素是一个128位的小数组，每个rank的对应点相累加，
// 累加完后降回到原始类型，输出给output。因为output是本地地址，每个rank都会进行这个操作，得到的output也会完全一致，不需要额外分发。
template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

// <NT> __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)指明线程块尺寸和资源占用的约束信息。
// maxThreadsPerBlock：最大允许的线程数是512，而这里allreduce函数默认的也是512。
// minBlocksPerMultiprocessor：每个SM必须能同时驻留的最小线程块数。不确定这里的用意？
// multi_gpu_barrier是为了多轮通信下的多个GPU的进展同步，比如有两轮通信，当前GPU完成了第一轮会马上开始准备第二轮，在第二轮计算前，发现其他GPU还在第一轮通信中，会进行等待。
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

// <NT> RingAllReduce的实现版本：ScatterReduce + Allgather
// A[123456] B[abcdef] C[ijklmn] 数据被分成三部分: A[12,34,56] B[ab,cd,ef] C[ij,kl,mn]
// A的part是0号，B的是1号，C是2号. 做allreduce后，A进行针对0号part得到[12abij,34,56], B[ab,34cdkl,ef], C[ij,kl,56efmn]，每个rank都有一个part的完整数据，
// 然后allgather，A需要从B拿到1号part，从C拿到2号part，其他rank类似。
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) cross_device_reduce_2stage(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // <NT> 按rank数量均等分块，part是一块的数据量，start是每块的起始点。
  // largest_part：如17份数据8个gpu，则有7个part为2，则有1个part为3. largest_part = (17+2) % 8 = 3
  int part = size / ngpus;  
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  // <NT> rank3, 对应target会是34567012
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

  // <NT> 所有rank的同步信号
  RankSignals sg_;
  // <NT> first是外部输入的显存地址指针，second是对应来自所有rank的peer指针，peer指针指向共享内存。
  // 
  // Stores an map from a pointer to its peer pointters from all ranks.
  std::unordered_map<void*, RankData*> buffers_;
  // <NT> 当前rank自身的同步信号
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

  // <NT> 将capture时调用allreduce接口的输入的tensor的指针都保存在graph_unreg_buffers_，在结束capture后调用该函数，通过cudaIpcGetMemHandle将其设置为共享内存。
  //  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR 可以得到 ptr 指向的内存块的首地址 。
  //  (内存需要是cudaMallocManaged申请的统一虚拟地址UVA 或 pytorch Tensor申请的显存(pytorch有封装相关信息)， 而不能是cudaMalloc申请的)。
  //  比如用cudaMallocManaged申请了一大块内存，用 base_ptr 指定，然后基于地址偏移将内存划分成多块，充当多个tensor，
  //  用每个块的首地址通过CU_POINTER_ATTRIBUTE_RANGE_START_ADDR都可以重新拿到偏移前的 base_ptr。
  //
  // handles存放基于首地址做内存共享得到的句柄，就是共享内存的首地址，偏移地址 - 首地址 = offset。共享内存的首地址加上偏移量就能拿到每个tensor在共享内存上的地址。
  // 因为不是所有tensor都来自同一块大内存，所以graph_unreg_buffers_里首地址也会有很多个，个别元素是同一块内存不同offset，个别元素是不同内存块。
  // 
  // 调用链路：python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
  //     capture -> (final) register_graph_buffers -> ops.get_graph_buffer_ipc_meta(当前函数) -> ops.register_graph_buffers
  //     即在capture结束后调用，capture中会通过下面 allreduce 的 status == cudaStreamCaptureStatusActive，将capture中输入的tensor全部加入到graph_unreg_buffers_中。
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
   * <NT> d_rank_data_base_ 是构造时传入的普通显存地址，专门用于当前rank的普通指针到所有rank的共享指针的映射关系。
   * (也参考python/sglang/srt/distributed/device_communicators/custom_all_reduce.py：create_shared_buffer， 当前rank留存的是普通指针，
   * 其他rank的需要用cudaIpcGetMemHandle/cudaIpcOpenMemHandle额外处理)
   * 输入参数 ptrs ，是 custom_all_reduce.py的buffer_ptrs，即是需要通信的目标数据的地址，一个小数组包含所有rank的共享内存地址，所以使用rank_data来存放这些指针数组。
   * 因为这个映射关系只需要本地可见，所以用torch.empty申请空间即可。
   * 
   * buffers_的first是通信用的目标数据的当前rank的内存地址，second是目标通信数据的所有rank的共享内存地址(包含了自身的)。
   * 实际使用时，输入当前rank的内存地址，就可以通过buffers_取得其对应的其他rank的对等的内存地址。直接基于这些内存进行规约操作即可。
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
    // <NT> 问：d_rank_data_base_在cuda graph的replay阶段，如何使用？
    // 答：因为下面allreduce函数的流程在capture中已经被固化，所以不会再显式地运行。再replay时会严格按照capture中的流程进行运行。
    // graph_unreg_buffers_的数量也已经被捕获到，所以在replay时，仍然可以看成是走if (status == cudaStreamCaptureStatusActive)的内容，
    // 所以到了特定的调用位置，会有特定的graph_unreg_buffers_.size()，通过ptrs = d_rank_data_base_ + graph_unreg_buffers_.size()，
    // 能够准确拿到特定调用位置的输入tensor对应的地址。
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
    // <NT> 当处于cuda graph正在capture中时，将外部提供的输入tensor加入到graph_unreg_buffers_中。
    // rank_data 存放的是目标数据 从 当前rank的显存指针 映射到 所有rank的地址数组 的映射关系, 针对cuda graph直接用graph_unreg_buffers_.size()充当偏移量。
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
    // <NT> 0是共享内存大小，指该kernel不需要动态分配共享内存，但不排除里面静态使用共享内存。
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
