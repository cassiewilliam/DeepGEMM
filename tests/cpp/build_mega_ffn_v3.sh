#!/usr/bin/env bash
# Build tests/cpp/test_sm100_fp8_mega_ffn_v3.cu (Per-Tensor FP8 variant)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${CUDA_HOME}/bin/nvcc"

if [[ ! -x "${NVCC}" ]]; then
    echo "nvcc not found at ${NVCC}. Set CUDA_HOME."
    exit 1
fi

if [[ -z "${CUTLASS_HOME:-}" ]]; then
    if [[ -d "${REPO_ROOT}/third-party/cutlass/include/cute" ]]; then
        CUTLASS_HOME="${REPO_ROOT}/third-party/cutlass"
    else
        echo "Please set CUTLASS_HOME"
        exit 1
    fi
fi

echo "[build_mega_ffn_v3] CUDA_HOME=${CUDA_HOME} CUTLASS_HOME=${CUTLASS_HOME}"

OUT="${SCRIPT_DIR}/test_mega_ffn_v3"
SRC="${SCRIPT_DIR}/test_sm100_fp8_mega_ffn_v3.cu"

INCS=(
    "-I${REPO_ROOT}/deep_gemm/include"
    "-I${CUTLASS_HOME}/include"
    "-I${CUDA_HOME}/include"
)

FLAGS=(
    -std=c++17
    -O3
    -lineinfo
    -gencode=arch=compute_100a,code=sm_100a
    -Xcompiler=-fPIC
    -Xcompiler=-Wno-psabi
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -DCUTE_USE_PACKED_TUPLE=1
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
)

if [[ "${TRACE:-0}" == "1" ]]; then
    FLAGS+=(-DMEGA_FFN_TRACE=1)
fi
if [[ -n "${STAGES:-}" ]];      then FLAGS+=(-DMFFN_STAGES=${STAGES}); fi
if [[ -n "${EPI_THREADS:-}" ]]; then FLAGS+=(-DMFFN_EPI_THREADS=${EPI_THREADS}); fi
if [[ -n "${CLUSTER_DIM:-}" ]]; then FLAGS+=(-DMFFN_CLUSTER_DIM=${CLUSTER_DIM}); fi
if [[ -n "${L2_K_SPLIT:-}" ]];  then FLAGS+=(-DMFFN_L2_K_SPLIT=${L2_K_SPLIT}); fi

LIBS=(
    "-L${CUDA_HOME}/lib64"
    -lcuda
    -lcudart
)

set -x
"${NVCC}" "${FLAGS[@]}" "${INCS[@]}" "${SRC}" "${LIBS[@]}" -o "${OUT}"
set +x
echo "[build_mega_ffn_v3] built: ${OUT}"
