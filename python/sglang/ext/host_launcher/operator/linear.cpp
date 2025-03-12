
#include "linear.h"
#include <cstring>
#include <thread>
#include <vector>

#ifdef ENABLE_AVX512
#include <immintrin.h>
#endif

Linear::Linear(LinearConfig config) {
    config_ = config;
    flag_ = 1;

    num_thread_ = 16;
    thread_pool_.CreateThreads(num_thread_);
}

Linear::~Linear() {}

// ����BF16��������
typedef uint16_t bf16;

// �����ȸ�����תBF16
bf16 float_to_bf16(float f) {
    uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    return static_cast<bf16>(u >> 16);
}

// BF16ת�����ȸ�����
float bf16_to_float(bf16 b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return *reinterpret_cast<float*>(&u);
}

#ifdef ENABLE_AVX512
void GemmBf16TbSplitN(int M, int N, int N_start, int N_end, int K, const bf16* A, const bf16* B, const bf16* bias, bf16* C) {
    int k;
    for (int i = 0; i < M; ++i) {
        for (int j = N_start; j < N_end; j++) {
            __m512 sum = _mm512_setzero_ps();
            for (k = 0; k < K-31; k+=32) {     // һ��32��16λ=512λ
                __m512i ai = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(A + i * K + k));  // ���ڴ��ȡ����������
                __m512bh a_bh = reinterpret_cast<__m512bh>(ai);                                    // ����ǿת
                __m512i bi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + j * K + k));
                __m512bh b_bh = reinterpret_cast<__m512bh>(bi);

                sum = _mm512_dpbf16_ps(sum, a_bh, b_bh);  // 32��bf16�ĵ�����㣬a_bh��b_bh������õ�16��fp32����16��fp32��sum��ӡ�cat /proc/cpuinfo ��Ҫ�� avx512_bf16�Ĳ�֧��
            }
            // float *sum_ptr = (float*)&sum;
            // float sum2 = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3] + sum_ptr[4] + sum_ptr[5] + sum_ptr[6] + sum_ptr[7] + 
            //              sum_ptr[8] + sum_ptr[9] + sum_ptr[10] + sum_ptr[11] + sum_ptr[12] + sum_ptr[13] + sum_ptr[14] + sum_ptr[15];
            float sum2 = _mm512_reduce_add_ps(sum);
            for (k; k < K; k++) {
                float a = bf16_to_float(A[i * K + k]);
                float b = bf16_to_float(B[j * K + k]);
                sum2 += a * b;
            }
            C[i * N + j] = float_to_bf16(sum2);
        }
    }
    if (bias != 0) {
        for (int i = 0; i < M; ++i) {
            for (int j = N_start; j < N_end; j++) {
                float a = bf16_to_float(C[i * N + j]);
                float b = bf16_to_float(bias[j]);
                C[i * N + j] = float_to_bf16(a+b);
            }
        }
    }
}

void GemmBf16Tb(int M, int N, int K, const bf16* A, const bf16* B, const bf16* bias, bf16* C) {
    GemmBf16TbSplitN(M, N, 0, N, K, A, B, bias, C);
}
#endif

void Linear::forward(int qlen, const void* input, const void* weight, const void* bias, void* output) {
    // printf("Linear::forward: %d, %d, %d.\n", qlen, config_.input_size, config_.output_size);
    printf("c");
    bf16 *A = (bf16 *)input;
    const bf16 *B = (const bf16 *)weight;
    const bf16 *bias_bf = (const bf16 *)bias;
    bf16 *C = (bf16 *)output;
    
    int M = qlen;
    int N = config_.output_size;
    int K = config_.input_size;
    /////////////////////////////////////
    // static int st = 0;
    // st++;
    // printf("st: %d.\n", st);
    // if (M == 6 && config_.input_size == 1536 && config_.output_size == 17920) {
    //     for (int i=0; i<M; i++) {
    //         for (int j=0; j<K; j++) {
    //             A_bf16[i * K + j] = float_to_bf16(fixed_a2[i * K + j]);
    //             A[i * K + j] = A_bf16[i * K + j];
    //         }
    //     }
    // }
    ////////////////////////////////////
#ifdef ENABLE_AVX512
    // printf("schema 1.\n");
    auto func = [&](const uint32_t n_start, const uint32_t n_end) {
        GemmBf16TbSplitN(M, N, n_start, n_end, K, A, B, bias_bf, C);
    };

    std::vector<std::future<void>> res;
    int chunk_size = N / num_thread_;  // ��17��16�̣߳�17/16==1, ǰ15���̴߳���1�������һ���̴߳���ʣ�µ�2��
    for (int i=0; i<num_thread_-1; i++) {
        res.push_back(thread_pool_.TaskEnqueue(func, i*chunk_size, (i+1)*chunk_size));
    }
    res.push_back(thread_pool_.TaskEnqueue(func, (num_thread_-1)*chunk_size, N));

    for (int i=0; i<num_thread_; i++)
        res[i].wait();

    // GemmBf16Tb(M,N,K, A, B, bias, C);
#else
    // printf("schema 2.\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = bf16_to_float(A[i * K + k]);
                float b = bf16_to_float(B[j * K + k]);
                sum += a * b;
            }
            if (bias_bf != 0)
                sum += bf16_to_float(bias_bf[j]);
            C[i * N + j] = float_to_bf16(sum);
        }
    }
#endif
    // if (1) {
    //     printf("A:\n");
    //     for (int i=0; i<M; i++) {
    //         for (int j=0; j<K; j++) {
    //             if (j%10 == 0)
    //                 printf("\n");
    //             printf("%f, ", bf16_to_float(A[i * K + j]));
    //         }
    //     }
    //     // if (st == 1) {
    //     //     printf("\nB:\n");
    //     //     for (int i=0; i<N; i++) {
    //     //         for (int j=0; j<K; j++) {
    //     //             if (j%10 == 0)
    //     //                 printf("\n");
    //     //             printf("%f, ", bf16_to_float(B[i * K + j]));
    //     //         }
    //     //     }            
    //     // }
    //     printf("\nC:\n");
    //     for (int i=0; i<M; i++) {
    //         for (int j=0; j<N; j++) {
    //             if (j%10 == 0)
    //                 printf("\n");
    //             printf("%f, ", bf16_to_float(C[i * N + j]));
    //         }
    //     }        
    // }
}

void Linear::forward_pure(int qlen, intptr_t input, intptr_t weight, intptr_t bias, intptr_t output, int flag) {
    flag_ = flag;
    forward(qlen, (const void *)input, (const void *)weight, (const void *)bias, (void *)output);
}