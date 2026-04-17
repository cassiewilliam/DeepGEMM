// =====================================================================================
// tests/cpp/test_ffn_cublas_baseline.cu
//
// Qwen3-0.6B FFN cuBLAS baseline：W1(gate) + W3(up) + SiLU + W2 三段不融合，
// 作为 MegaFFN (FP8 fused) 的性能对照。
//
// 变体：
//   - BF16：cublasGemmEx(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F compute)
//
// 维度（Qwen3-0.6B）：
//   hidden (H)       = 1024
//   intermediate (I) = 3072
//
// 计算：
//   gate[M,I] = X[M,H] @ W1[I,H]^T
//   up  [M,I] = X[M,H] @ W3[I,H]^T
//   interm    = SiLU(gate) * up   // elementwise
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

// SiLU(x) * y elementwise，in-place on gate buffer
__global__ void silu_mul_kernel(__nv_bfloat16* __restrict__ gate,
                                const __nv_bfloat16* __restrict__ up,
                                uint32_t n_elems) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    float s = g / (1.0f + expf(-g));           // SiLU
    gate[idx] = __float2bfloat16(s * u);
}

int main(int argc, char** argv) {
    uint32_t M         = argc > 1 ? static_cast<uint32_t>(std::atoi(argv[1])) : 32;
    uint32_t num_iters = argc > 2 ? static_cast<uint32_t>(std::atoi(argv[2])) : 100;

    std::printf("=== cuBLAS BF16 FFN baseline  M=%u  H=%u  I=%u ===\n", M, H, I);

    // --- Random init on host ---
    std::mt19937 rng(0xABCDu);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> hX(static_cast<size_t>(M) * H);
    std::vector<__nv_bfloat16> hW1(static_cast<size_t>(I) * H);
    std::vector<__nv_bfloat16> hW3(static_cast<size_t>(I) * H);
    std::vector<__nv_bfloat16> hW2(static_cast<size_t>(H) * I);
    for (auto& v : hX)  v = __float2bfloat16(dist(rng));
    for (auto& v : hW1) v = __float2bfloat16(dist(rng));
    for (auto& v : hW3) v = __float2bfloat16(dist(rng));
    for (auto& v : hW2) v = __float2bfloat16(dist(rng));

    __nv_bfloat16 *dX, *dW1, *dW3, *dW2, *dGate, *dUp, *dY;
    CUDA_CHECK(cudaMalloc(&dX,   sizeof(__nv_bfloat16) * M * H));
    CUDA_CHECK(cudaMalloc(&dW1,  sizeof(__nv_bfloat16) * I * H));
    CUDA_CHECK(cudaMalloc(&dW3,  sizeof(__nv_bfloat16) * I * H));
    CUDA_CHECK(cudaMalloc(&dW2,  sizeof(__nv_bfloat16) * H * I));
    CUDA_CHECK(cudaMalloc(&dGate,sizeof(__nv_bfloat16) * M * I));
    CUDA_CHECK(cudaMalloc(&dUp,  sizeof(__nv_bfloat16) * M * I));
    CUDA_CHECK(cudaMalloc(&dY,   sizeof(__nv_bfloat16) * M * H));

    CUDA_CHECK(cudaMemcpy(dX,  hX.data(),  sizeof(__nv_bfloat16) * M * H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, hW1.data(), sizeof(__nv_bfloat16) * I * H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW3, hW3.data(), sizeof(__nv_bfloat16) * I * H, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, hW2.data(), sizeof(__nv_bfloat16) * H * I, cudaMemcpyHostToDevice));

    cublasHandle_t cb;
    CUBLAS_CHECK(cublasCreate(&cb));
    // 确保使用 Tensor Core
    CUBLAS_CHECK(cublasSetMathMode(cb, CUBLAS_TENSOR_OP_MATH));

    const float alpha_f = 1.0f, beta_f = 0.0f;

    // cuBLAS 是 column-major。我们按 row-major 解读成：
    //   C[m,n] = A[m,k] * B[k,n]  =>  在 cuBLAS 视角下 C^T = B^T * A^T
    //   即调用 gemm(op=N, op=N, n, m, k, B, n, A, k, C, n).
    // 我们用常规写法：把 A 当成 [K,M] col-major, B 当成 [N,K] col-major, C 当成 [N,M] col-major.
    // 等价于 row-major 的 A[M,K], B[K,N] -> C[M,N].
    //
    // 为了 clarity，我们直接用 cublasGemmEx 做：
    //   gate = X @ W1^T  (X:[M,H], W1:[I,H])
    //     row-major: gate[M,I] = X[M,H] * W1[I,H]^T = X * (W1^T)^T
    //     column-major 等价: gate_cm[I,M] = W1_cm[H,I]^T^T * X_cm[H,M]^T^T ... (略)
    //   简单做法：col-major 视角
    //     A = W1 (col-major [H,I])，opA = N
    //     B = X  (col-major [H,M])，opB = T
    //     C = gate (col-major [I,M])
    //   由 cuBLAS 语义：C = op(A) * op(B) = W1^T * X  -> 维度 [I,H]*[H,M] = [I,M] ✓
    //   gate[i,m] = sum_h W1[h,i] * X[h,m]
    //             = sum_h W1_row[i,h] * X_row[m,h]   (row-major 下 W1[i,h]、X[m,h])
    //             = (X @ W1^T)[m,i]                   ✓
    //
    // 综上：
    //   op(A)=T,  op(B)=N,  A=W1 (ld=H), B=X (ld=H), C=gate (ld=I), m=I, n=M, k=H.

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
        // gate = X @ W1^T  : opA=T, opB=N, A=W1[H,I]col, B=X[H,M]col, C=gate[I,M]col
        gemm_bf16(CUBLAS_OP_T, CUBLAS_OP_N, I, M, H,
                  dW1, H, dX, H, dGate, I);
        // up   = X @ W3^T
        gemm_bf16(CUBLAS_OP_T, CUBLAS_OP_N, I, M, H,
                  dW3, H, dX, H, dUp, I);
        // interm = SiLU(gate) * up  (in-place on gate)
        int n_elems = static_cast<int>(M) * I;
        int blk = 256;
        silu_mul_kernel<<<(n_elems + blk - 1) / blk, blk>>>(dGate, dUp, n_elems);
        // Y = interm @ W2^T  : opA=T, opB=N, A=W2[I,H]col, B=gate[I,M]col, C=Y[H,M]col
        gemm_bf16(CUBLAS_OP_T, CUBLAS_OP_N, H, M, I,
                  dW2, I, dGate, I, dY, H);
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

    cudaFree(dX); cudaFree(dW1); cudaFree(dW3); cudaFree(dW2);
    cudaFree(dGate); cudaFree(dUp); cudaFree(dY);
    cublasDestroy(cb);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
