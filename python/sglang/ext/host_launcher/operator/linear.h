/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:00
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_LINEAR_H
#define CPUINFER_OPERATOR_LINEAR_H

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

#endif