#!/usr/bin/env bash
# Build cuBLAS BF16 FFN baseline
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${CUDA_HOME}/bin/nvcc"

SRC="${SCRIPT_DIR}/test_ffn_cublas_baseline.cu"
OUT="${SCRIPT_DIR}/test_ffn_cublas_baseline"

set -x
"${NVCC}" -std=c++17 -O3 \
    -gencode=arch=compute_100a,code=sm_100a \
    -Xcompiler=-fPIC \
    --expt-relaxed-constexpr \
    "${SRC}" \
    -L"${CUDA_HOME}/lib64" -lcuda -lcudart -lcublas \
    -o "${OUT}"
set +x
echo "[build_ffn_cublas_baseline] built: ${OUT}"
