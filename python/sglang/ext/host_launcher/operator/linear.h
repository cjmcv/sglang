
#ifndef OFFLOAD_LINEAR_H
#define OFFLOAD_LINEAR_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include "../thread_pool.hpp"

struct LinearConfig {
    int input_size;
    int output_size;

    LinearConfig() {}
    LinearConfig(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {}
};

class Linear {
public:
    Linear(LinearConfig);
    ~Linear();

    void forward_pure(int qlen, intptr_t input, intptr_t weight, intptr_t bias, intptr_t output, int flag);
    void forward(int qlen, const void* input, const void* weight, const void* bias, void* output);

private:
    LinearConfig config_;
    int flag_;

    int num_thread_;
    pai::thread::ThreadPool thread_pool_;
};

#endif // OFFLOAD_LINEAR_H