
#include "linear.h"

Linear::Linear(LinearConfig config) {
    config_ = config;
}

Linear::~Linear() {}

// 定义BF16数据类型
typedef uint16_t bf16;

// 单精度浮点数转BF16
bf16 float_to_bf16(float f) {
    uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    return static_cast<bf16>(u >> 16);
}

// BF16转单精度浮点数
float bf16_to_float(bf16 b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return *reinterpret_cast<float*>(&u);
}

void Linear::forward(int qlen, const void* input, const void* weight, void* output) {
    printf("Linear::forward: %d, %d, %d.\n", qlen, config_.input_size, config_.output_size);
    const bf16 *A = (const bf16 *)input;
    const bf16 *B = (const bf16 *)weight;
    bf16 *C = (bf16 *)output;

    int M = qlen;
    int N = config_.output_size;
    int K = config_.input_size;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = bf16_to_float(A[i * K + k]);
                float b = bf16_to_float(B[j * K + k]);
                sum += a * b;
            }
            C[i * N + j] = float_to_bf16(sum);
        }
    }

    // if (M == 6) {
    //     printf("inside.\n");
    //     for (int i=0; i<M; i++) {
    //         for (int j=0; j<N; j++) {
    //             if (j%5 == 0)
    //                 printf("\n");
    //             printf("%f, ", bf16_to_float(C[i * N + j]));
    //         }
    //     }        
    // }
}