
#ifndef OFFLOAD_LINEAR_H
#define OFFLOAD_LINEAR_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include "../thread_pool.hpp"

struct LinearConfig {
    int mode;

    LinearConfig() {}
    LinearConfig(int mode): mode(mode) {}
};

class Linear {
public:
    Linear(LinearConfig);
    ~Linear();

    void forward_pure(int M, int N, int K, intptr_t input, intptr_t weight, intptr_t bias, intptr_t output, int flag);
    void forward(int M, int N, int K, const void* input, const void* weight, const void* bias, void* output);

    void ForwardW4A16WithFp32(int M, int N, int K, const void* input, const void* weight, const void* bias, void* output);
    void UnpackW4ToFp32(const void *input_weight, int rows, int cols, const void *qzeros, const void *scales, void* output_weight);

private:
    std::vector<int> UnpackInt32ToInt4(int int32_value);
    
private:
    LinearConfig config_;
    int flag_;

    int num_thread_;
    pai::thread::ThreadPool thread_pool_;

    float *table_f32_f16_;
    float *At_;
    int At_len_;
    float *Ct_;
    int Ct_len_;
};

#endif // OFFLOAD_LINEAR_H