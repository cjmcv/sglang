
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

    table_f32_f16_ = nullptr;
    At_ = nullptr;
    At_len_ = 0;
    Ct_ = nullptr;
    Ct_len_ = 0;
}

Linear::~Linear() {
    if (table_f32_f16_ != nullptr) {
        delete[] table_f32_f16_;
        delete[] At_;
        delete[] Ct_;        
    }
}

// 定义BF16数据类型
typedef uint16_t bf16;
typedef uint16_t fp16;

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

static float ConvertHalfToFp32(fp16 h) {

#ifdef __F16C__
    return _cvtsh_ss(h);
#else
    uint32_t sign = ((h >> 15) & 1);
    uint32_t exp = ((h >> 10) & 0x1f);
    uint32_t mantissa = (h & 0x3ff);
    unsigned f = 0;

    if (exp > 0 && exp < 31) {
        // normal
        exp += 112;
        f = (sign << 31) | (exp << 23) | (mantissa << 13);
    }
    else if (exp == 0) {
        if (mantissa) {
            // subnormal
            exp += 113;
            while ((mantissa & (1 << 10)) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3ff;
            f = (sign << 31) | (exp << 23) | (mantissa << 13);
        }
        else {
            // sign-preserving zero
            f = (sign << 31);
        }
    }
    else if (exp == 31) {
        if (mantissa) {
            f = 0x7fffffff; // not a number
        }
        else {
            f = (0xff << 23) | (sign << 31); //  inf
        }
    }
    float flt;
    std::memcpy(&flt, &f, sizeof(flt));
    return flt;
#endif
}

static fp16 ConvertFp32ToHalf(float flt) {
#ifdef __F16C__
    return _cvtss_sh(flt, 0);
#else
    uint32_t s = *reinterpret_cast<uint32_t*>(&flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int16_t exp = uint16_t(((s >> 23) & 0xff) - 127);
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
        // sign-preserving zero
        return sign;
    }

    if (exp > 15) {
        if (exp == 128 && mantissa) {
            // not a number
            u = 0x7fff;
        } else {
            // overflow to infinity
            u = sign | 0x7c00;
        }
        return u;
    }

    int sticky_bit = 0;

    if (exp >= -14) {
        // normal fp32 to normal fp16
        exp = uint16_t(exp + uint16_t(15));
        u = uint16_t(((exp & 0x1f) << 10));
        u = uint16_t(u | (mantissa >> 13));
    } else {
        // normal single-precision to subnormal half_t-precision representation
        int rshift = (-14 - exp);
        if (rshift < 32) {
            mantissa |= (1 << 23);

            sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);

            mantissa = (mantissa >> rshift);
            u = (uint16_t(mantissa >> 13) & 0x3ff);
        } else {
            mantissa = 0;
            u = 0;
        }
    }

    // round to nearest even
    int round_bit = ((mantissa >> 12) & 1);
    sticky_bit |= ((mantissa & ((1 << 12) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
        u = uint16_t(u + 1);
    }

    u |= sign;

    return u;
#endif
}

#ifdef ENABLE_AVX512
void GemmBf16TbSplitN(int M, int N, int N_start, int N_end, int K, const bf16* A, const bf16* B, const bf16* bias, bf16* C) {
    int k;
    for (int i = 0; i < M; ++i) {
        for (int j = N_start; j < N_end; j++) {
            __m512 sum = _mm512_setzero_ps();
            for (k = 0; k < K-31; k+=32) {     // 一次32个16位=512位
                __m512i ai = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(A + i * K + k));  // 按内存读取，不管类型
                __m512bh a_bh = reinterpret_cast<__m512bh>(ai);                                    // 类型强转
                __m512i bi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(B + j * K + k));
                __m512bh b_bh = reinterpret_cast<__m512bh>(bi);

                sum = _mm512_dpbf16_ps(sum, a_bh, b_bh);  // 32个bf16的点积运算，a_bh和b_bh点积，得到16个fp32，与16个fp32的sum相加。cat /proc/cpuinfo 需要有 avx512_bf16的才支持
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

// def unpack_int32_to_int4(self, int32_value):
//     # 用于存储拆分后的 8 个 int4 值
//     int4_values = []
//     # 循环 8 次，每次提取一个 int4
//     for i in range(8):
//         # 右移操作，将当前要提取的 int4 移动到最低 4 位
//         shifted = int32_value >> (i * 4)
//         # 通过按位与操作，提取最低 4 位
//         int4_value = shifted & 0xF
//         int4_values.append(int4_value)
//     return int4_values

std::vector<int> Linear::UnpackInt32ToInt4(int int32_value) {
    std::vector<int> vec;
    for (int i=0; i<8; i++) {
        int shifted = int32_value >> (i * 4);
        int int4_value = shifted & 0xF;
        vec.push_back(int4_value);
    }
    return vec;
}

void Linear::UnpackW4ToFp32(const void *qweight, int rows, int cols, const void *qzeros, const void *scales, void* weight_fp32) {
    // # # 某层例子：k=8960，n=1536
    // # # 权重一个元素打包同列连续8行的元素，如qw[i,j]=w[i*8+(0~7), j]
    // # # layer.qweight.shape torch.Size([1120, 1536])  1120 = 8960 // 8  int32，原维度[8960, 1536], 普通线性层应该是[1536, 8960]
    // # # layer.scales.shape torch.Size([70, 1536])     70 = 8960 // 128 = 8960 / 8 / 16   fp16 ，原维度[70, 1536], 分组按k维度划分
    // # # layer.qzeros.shape torch.Size([70, 192])      192 = 1536 // 8，  int32，原维度[70, 1536]
    // # # layer.g_idx.shape torch.Size([8960])
    // # # in.shape: torch.Size([m, 8960])
    // # # out.shape: torch.Size([m, 1536])
    // #
    // # for k in range(qweight_cpu.shape[0]):
    // #     print(k)
    // #     for j in range(qweight_cpu.shape[1]):
    // #         int4_vec = self.unpack_int32_to_int4(qweight_cpu[k, j])
    // #         zero_vec = self.unpack_int32_to_int4(layer.qzeros[k//16, j//8]) # 16 = 128/8
    // #         for ki in range(8):
    // #             zero = zero_vec[j%8]
    // #             scale = layer.scales[k//16, j]
    // #             # self.weight_fp16[8*k+ki, j] = (int4_vec[ki] - zero - 1) * scale
    // #             self.weight_fp16[j, 8*k+ki] = (int4_vec[ki] - zero - 1) * scale # 额外转置，用于linear

    int *in = (int *)qweight;
    int *z = (int *)qzeros;
    fp16 *s = (fp16 *)scales;
    float *out = (float *)weight_fp32;
    //////////////////////////////////////////
    // for (int k=0; k<rows; k++) {
    //     // printf("k: %d.\n", k);
    //     for (int j=0; j<cols; j++) {
    //         std::vector<int> int4_vec = UnpackInt32ToInt4(in[k*cols + j]);
    //         std::vector<int> zero_vec = UnpackInt32ToInt4(z[(k/16) * (cols/8) + j/8]); // k/16 = (k*8)/128
    //         for (int ki=0; ki<8; ki++) {
    //             float scale = ConvertHalfToFp32(s[(k/16) * cols + j]);
    //             int zero = zero_vec[j%8];
    //             // https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_marlin.py#194, zero解压时会加1 !!!
    //             float of = (int4_vec[ki] - zero - 1) * scale;
    //             // out[(8*k+ki)*cols + j] = of; // 非转置
    //             out[j*rows*8 + (8*k+ki)] = of;    // 额外转置，用于linear
    //         }
    //     }
    // }
    //////////////////////////////////////
    int N = cols;
    auto func = [&](const uint32_t n_start, const uint32_t n_end) {
        for (int k=0; k<rows; k++) {
            // printf("k: %d.\n", k);
            for (int j=n_start; j<n_end; j++) {
                std::vector<int> int4_vec = UnpackInt32ToInt4(in[k*cols + j]);
                std::vector<int> zero_vec = UnpackInt32ToInt4(z[(k/16) * (cols/8) + j/8]); // k/16 = (k*8)/128
                for (int ki=0; ki<8; ki++) {
                    float scale = ConvertHalfToFp32(s[(k/16) * cols + j]);
                    int zero = zero_vec[j%8];
                    // https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_marlin.py#194, zero解压时会加1 !!!
                    float of = (int4_vec[ki] - zero - 1) * scale;
                    // out[(8*k+ki)*cols + j] = of; // 非转置
                    out[j*rows*8 + (8*k+ki)] = of;    // 额外转置，用于linear
                }
            }
        }
    };
    std::vector<std::future<void>> res;
    int chunk_size = N / num_thread_;  // 如17个16线程，17/16==1, 前15个线程处理1个，最后一个线程处理剩下的2个
    for (int i=0; i<num_thread_-1; i++) {
        res.push_back(thread_pool_.TaskEnqueue(func, i*chunk_size, (i+1)*chunk_size));
    }
    res.push_back(thread_pool_.TaskEnqueue(func, (num_thread_-1)*chunk_size, N));
    for (int i=0; i<num_thread_; i++)
        res[i].wait();
    ////////////////////////////////////

    if (table_f32_f16_ == nullptr) {
        printf("table_f32_f16_ == nullptr.\n");
        table_f32_f16_ = new float[1 << 16];
        for (int i = 0; i < (1 << 16); ++i) {
            table_f32_f16_[i] = ConvertHalfToFp32(i);
        }
    }
}

#ifdef ENABLE_AVX512
void LinearAvx512Fp32SplitN(int M, int N, int N_start, int N_end, int K, 
                            const float* A, const float* B, const float* bias, float* C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = N_start; j < N_end; ++j) {
            __m512 vsum = _mm512_setzero_ps();
            for (k = 0; k < K-15; k += 16) {
                __m512 a = _mm512_loadu_ps(A + i*K + k);
                __m512 b = _mm512_loadu_ps(B + j*K + k);
                vsum = _mm512_fmadd_ps(a, b, vsum);
            }
            float sum = 0;
            for (k; k < K; k++) {
                sum += A[i*K + k] * B[j*K + k];
            }
            for (int t=0; t<16; t++) {
                sum += vsum[t];                
            }
            C[i*N + j] = sum;
        }
    }    
}
#endif

void Linear::ForwardW4A16WithFp32(int M, int N, int K, const void* input, const void* weight, const void* bias, void* output) {
    fp16 *A = (fp16 *)input;
    const float *B = (const float *)weight;
    fp16 *C = (fp16 *)output;
    
    // int M = qlen;
    // int N = config_.output_size;
    // int K = config_.input_size;
#ifdef ENABLE_AVX512
    if (At_len_ < M * K) {
        At_len_ = M * K;
        if (At_ != nullptr) {
            delete[] At_;
        }
        At_ = new float[At_len_];
    }
    if (Ct_len_ < M * N) {
        Ct_len_ = M * N;
        if (Ct_ != nullptr) {
            delete[] Ct_;    
        }
        Ct_ = new float[Ct_len_];
    }
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            At_[i*K + k] = table_f32_f16_[A[i*K + k]];

    /////////////////////////////////////
    auto func = [&](const uint32_t n_start, const uint32_t n_end) {
        LinearAvx512Fp32SplitN(M, N, n_start, n_end, K, At_, B, NULL, Ct_);
    };
    std::vector<std::future<void>> res;
    int chunk_size = N / num_thread_;  // 如17个16线程，17/16==1, 前15个线程处理1个，最后一个线程处理剩下的2个
    for (int i=0; i<num_thread_-1; i++) {
        res.push_back(thread_pool_.TaskEnqueue(func, i*chunk_size, (i+1)*chunk_size));
    }
    res.push_back(thread_pool_.TaskEnqueue(func, (num_thread_-1)*chunk_size, N));
    for (int i=0; i<num_thread_; i++)
        res[i].wait();

    // LinearAvx512Fp32SplitN(M, N, 0, N, K, At_, B, Ct_);
    /////////////////////////////////////

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i*N+j] = ConvertFp32ToHalf(Ct_[i*N+j]);
#else
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += ConvertHalfToFp32(A[i * K + k]) * B[j * K + k];
            }
            C[i * N + j] = ConvertFp32ToHalf(sum);
        }
    }
#endif
}

void Linear::forward(int M, int N, int K, const void* input, const void* weight, const void* bias, void* output) {
    if (config_.mode == 1)
        return ForwardW4A16WithFp32(M,N,K, input, weight, bias, output);

    // printf("Linear::forward: %d, %d, %d.\n", qlen, config_.input_size, config_.output_size);
    printf("c");
    bf16 *A = (bf16 *)input;
    const bf16 *B = (const bf16 *)weight;
    const bf16 *bias_bf = (const bf16 *)bias;
    bf16 *C = (bf16 *)output;
    
    // int M = qlen;
    // int N = config_.output_size;
    // int K = config_.input_size;
    ////////////////////////////////////
#ifdef ENABLE_AVX512
    // printf("schema 1.\n");
    auto func = [&](const uint32_t n_start, const uint32_t n_end) {
        GemmBf16TbSplitN(M, N, n_start, n_end, K, A, B, bias_bf, C);
    };

    std::vector<std::future<void>> res;
    int chunk_size = N / num_thread_;  // 如17个16线程，17/16==1, 前15个线程处理1个，最后一个线程处理剩下的2个
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
}

void Linear::forward_pure(int M, int N, int K, intptr_t input, intptr_t weight, intptr_t bias, intptr_t output, int flag) {
    flag_ = flag;
    forward(M,N,K, (const void *)input, (const void *)weight, (const void *)bias, (void *)output);
}