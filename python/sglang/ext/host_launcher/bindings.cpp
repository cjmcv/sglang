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
        int qlen;
        const void *input;
        const void *weight;
        void *output;
    };
    static void inner(void *args) {
        Args *args_ = (Args *)args;
        args_->linear->forward(args_->qlen, args_->input, args_->weight, args_->output);
    }
    static std::pair<intptr_t, intptr_t>
    warp4launch(Linear &linear, int qlen, intptr_t input, intptr_t weight, intptr_t output) {
        Args *args = new Args{nullptr, &linear, qlen, (const void *)input, (const void *)weight,
                              (void *)output};
        return std::make_pair((intptr_t)&inner, (intptr_t)args);
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
        .def(py::init([](int hidden_size, int intermediate_size) {
            return LinearConfig(hidden_size, intermediate_size);
        }));
    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())  // The input parameter type of this constructor is LinearConfig.
        .def("forward", &LinearBindings::warp4launch);
}