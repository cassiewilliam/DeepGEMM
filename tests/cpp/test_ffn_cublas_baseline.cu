// =====================================================================================
// tests/cpp/test_ffn_cublas_baseline.cu
//
// Qwen3-0.6B FFN cuBLAS baseline：W1(gate+up 合并) + SwiGLU + W2 三段不融合，
// 作为 MegaFFN (FP8 fused) 的性能对照。
//
// 变体：
//   - BF16：cublasGemmEx(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F compute)
//
// 维度（Qwen3-0.6B）：
//   hidden (H)       = 1024
//   intermediate (I) = 3072
//
// 计算（gate/up 合并为 W1[2I, H] 单次 GEMM，与 MegaFFN kernel 语义一致，避免两次 HBM 往返）：
//   gu[M, 2I] = X[M,H] @ W1[2I,H]^T           // 前 I 列 = gate, 后 I 列 = up
//   interm    = SiLU(gu[:, :I]) * gu[:, I:]   // SwiGLU
//   Y [M,H]   = interm[M,I] @ W2[H,I]^T
//
// 用法：
//   ./test_ffn_cublas_baseline [M=32] [num_iters=100]
//
// 编译参考 tests/cpp/build_ffn_cublas_baseline.sh
// =====================================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {                  \
    std::fprintf(stderr, "CUDA %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); } } while (0)

#define CUBLAS_CHECK(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) {    \
    std::fprintf(stderr, "cuBLAS %s:%d : %d\n", __FILE__, __LINE__, (int)s);              \
    std::exit(1); } } while (0)

constexpr uint32_t H = 1024;
constexpr uint32_t I = 3072;

// SwiGLU on fused gate/up buffer gu[M, 2I]，输出 interm[M, I]。
// 约定 gu[m, 0..I)   = gate, gu[m, I..2I) = up。
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ gu,
                              __nv_bfloat16* __restrict__ interm,
                              uint32_t M, uint32_t I) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n_elems = M * I;
    if (idx >= n_elems) return;
    uint32_t m = idx / I;
    uint32_t c = idx - m * I;
    float g = __bfloat162float(gu[m * 2 * I + c]);
    float u = __bfloat162float(gu[m * 2 * I + I + c]);
    float s = g / (1.0f + expf(-g));           // SiLU
    interm[idx] = __float2bfloat16(s * u);
}

int main(int argc, char** argv) {
    uint32_t M         = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 32;
    uint32_t num_iters = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 100;

    std::printf("=== cuBLAS BF16 FFN baseline  M=%u  H=%u  I=%u ===\n", M, H, I);

    // --- Random init on host ---
    std::mt19937 rng(0xABCDu);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    // W1 合并 gate+up 为 [2I, H] 的单个权重矩阵 (与 MegaFFN kernel 一致)
    std::vector<__nv_bfloat16> hX(static_cast<size_t>(M) * H);
    std::vector<__nv_bfloat16> hW1(static_cast<size_t>(2) * I * H);
    std::vector<__nv_bfloat16> hW2(static_cast<size_t>(H) * I);
    for (auto& v : hX)  v = __float2bfloat16(dist(rng));
    for (auto& v : hW1) v = __float2bfloat16(dist(rng));
    for (auto& v : hW2) v = __float2bfloat16(dist(rng));

    __nv_bfloat16 *dX, *dW1, *dW2, *dGU, *dInterm, *dY;
    CUDA_CHECK(cudaMalloc(&dX,     sizeof(__nv_bfloat16) * M * H));
    CUDA_CHECK(cudaMalloc(&dW1,    sizeof(__nv_bfloat16) * 2 * I * H));
    CUDA_CHECK(cudaMalloc(&dW2,    sizeof(__nv_bfloat16) * H * I));
    CUDA_CHECK(cudaMalloc(&dGU,    sizeof(__nv_bfloat16) * M * 2 * I));
    CUDA_CHECK(cudaMalloc(&dInterm,sizeof(__nv_bfloat16) * M * I));
    CUDA_CHECK(cudaMalloc(&dY,     sizeof(__nv_bfloat16) * M * H));

    CUDA_CHECK(cudaMemcpy(dX,  hX.data(),  sizeof(__nv_bfloat16) * M * H,         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, hW1.data(), sizeof(__nv_bfloat16) * 2 * I * H,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, hW2.data(), sizeof(__nv_bfloat16) * H * I,         cudaMemcpyHostToDevice));

    cublasHandle_t cb;
    CUBLAS_CHECK(cublasCreate(&cb));
    // 确保使用 Tensor Core
    CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_TENSOR_OP_MATH));

    const float alpha_f = 1.0f, beta_f = 0.0f;

    // cuBLAS 是 column-major；row-major 下的 C[M,N] = A[M,K] @ B[K,N]^T 等价于
    // col-major 下 opA=T, opB=N, A=B_cm[K,N], B=A_cm[K,M], C=C_cm[N,M], m=N, n=M, k=K。

    auto gemm_bf16 = [&](cublasOperation_t opA, cublasOperation_t opB,
                         int m, int n, int k,
                         const __nv_bfloat16* A, int lda,
                         const __nv_bfloat16* B, int ldb,
                         __nv_bfloat16* C, int ldc) {
        CUBLAS_CHECK(cublasGemmEx(cb, opA, opB, m, n, k,
            &alpha_f,
            A, CUDA_R_16BF, lda,
            B, CUDA_R_16BF, ldb,
            &beta_f,
            C, CUDA_R_16BF, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    auto launch_ffn = [&]() {
        // Linear1 (gate+up 合并): gu[M, 2I] = X[M, H] @ W1[2I, H]^T
        //   col-major: opA=T, opB=N, A=W1[H, 2I] (lda=H), B=X[H, M] (ldb=H), C=gu[2I, M] (ldc=2I)
        gemm_bf16(CUBLAS_OP_T, CUBLAS_OP_N, 2 * I, M, H,
                  dW1, H, dX, H, dGU, 2 * I);
        // SwiGLU: interm[M, I] = SiLU(gu[:, :I]) * gu[:, I:]
        int n_elems = static_cast<int>(M) * I;
        int blk = 256;
        swiglu_kernel<<<(n_elems + blk - 1) / blk, blk>>>(dGU, dInterm, M, I);
        // Linear2: Y[M, H] = interm[M, I] @ W2[H, I]^T
        gemm_bf16(CUBLAS_OP_T, CUBLAS_OP_N, H, M, I,
                  dW2, I, dInterm, I, dY, H);
    };

    // Warmup
    for (int i = 0; i < 20; ++ i) launch_ffn();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (uint32_t i = 0; i < num_iters; ++ i) launch_ffn();
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double avg_us = 1000.0 * ms / num_iters;
    // FLOPs: 2*M*H*I (W1) + 2*M*H*I (W3) + 2*M*I*H (W2)  = 6*M*H*I
    double flops = 6.0 * M * H * I / (avg_us * 1e-6);
    std::printf("[cublasBF16] M=%u  avg latency = %.3f us  (%.2f GFLOP/s @ BF16)\n",
                M, avg_us, flops * 1e-9);

    cudaFree(dX); cudaFree(dW1); cudaFree(dW2);
    cudaFree(dGU); cudaFree(dInterm); cudaFree(dY);
    cublasDestroy(cb);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
