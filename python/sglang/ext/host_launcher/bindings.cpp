#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "cuda_runtime.h"
#include "operator/linear.h"

namespace py = pybind11;
using namespace pybind11::literals;

class HostLauncher {
public:
    void hello(int i) {
         printf("hello: %d.\n", i);
    }
 
    void submit(std::pair<intptr_t, intptr_t> params) {
        void (*func)(void*) = (void (*)(void*))params.first;
        void* args = (void*)params.second;
        *((HostLauncher**)args) = this;
        func(args);
    }
 
    void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
        void (*func)(void*) = (void (*)(void*))params.first;
        void* args = (void*)params.second;
        *((HostLauncher**)args) = this;
        cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
    }
};

class LinearBindings {
public:
    struct Args {
        HostLauncher *cpuinfer;
        Linear *linear;
        int M;
        int N;
        int K;
        const void *input;
        const void *weight;
        const void *bias;
        void *output;
    };
    static void inner(void *args) {
        Args *args_ = (Args *)args;
        args_->linear->forward(args_->M, args_->N, args_->K, args_->input, args_->weight, args_->bias, args_->output);
    }
    static std::pair<intptr_t, intptr_t>
    warp4launch(Linear &linear, int M, int N, int K, intptr_t input, intptr_t weight, intptr_t bias, intptr_t output) {
        Args *args = new Args{nullptr, &linear, M,N,K, (const void *)input, (const void *)weight, (const void *)bias, (void *)output};
        return std::make_pair((intptr_t)&inner, (intptr_t)args);
    }
    ///////////////////////////////////////////////
    struct UnpackW4ToFp16Args {
        HostLauncher *cpuinfer;
        Linear *linear;
        const void *input_weight;
        int rows;
        int cols;
        const void *qzeros;
        const void *scales;
        void *output_weight;
    };
    static void unpack_inner(void *args) {
        UnpackW4ToFp16Args *args_ = (UnpackW4ToFp16Args *)args;
        args_->linear->UnpackW4ToFp32(args_->input_weight, args_->rows, args_->cols, args_->qzeros, args_->scales, args_->output_weight);
    }
    static std::pair<intptr_t, intptr_t>
    Warp4UnpackW4ToFp16(Linear &linear, intptr_t input_weight, int rows, int cols, intptr_t qzeros, intptr_t scales, intptr_t output_weight) {
        UnpackW4ToFp16Args *args = new UnpackW4ToFp16Args{nullptr, &linear, (const void *)input_weight, rows, cols, (const void *)qzeros, (const void *)scales, (void *)output_weight};
        return std::make_pair((intptr_t)&unpack_inner, (intptr_t)args);
    }
};


PYBIND11_MODULE(host_launcher, m) {
    py::class_<HostLauncher>(m, "HostLauncher")
        .def(py::init())  // constructor
        .def("hello", &HostLauncher::hello)
        .def("submit", &HostLauncher::submit)
        .def("submit_with_cuda_stream", &HostLauncher::submit_with_cuda_stream);
        
    auto linear_module = m.def_submodule("linear"); // submodule, use host_launcher.linear to use it.
    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int mode) {
            return LinearConfig(mode);
        }));
    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())  // The input parameter type of this constructor is LinearConfig.
        .def("forward", &LinearBindings::warp4launch)
        .def("forward_pure", &Linear::forward_pure)
        .def("UnpackW4ToFp32", &LinearBindings::Warp4UnpackW4ToFp16);
}