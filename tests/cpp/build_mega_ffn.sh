#!/usr/bin/env bash
# =====================================================================================
# tests/cpp/build_mega_ffn.sh
#
# 构建 tests/cpp/test_sm100_fp8_mega_ffn.cu —— 独立 C++ 编译入口
# 约束：B200 (sm_100a)，CUDA 12.8+，CUTLASS 头文件可访问。
#
# 使用方式：
#   export CUTLASS_HOME=/path/to/cutlass   # 必须指向包含 include/cute 的 CUTLASS 仓库
#   ./tests/cpp/build_mega_ffn.sh
#
# 或：
#   CUDA_HOME=/usr/local/cuda-12.8 CUTLASS_HOME=~/cutlass ./tests/cpp/build_mega_ffn.sh
# =====================================================================================
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
    # 尝试常见位置
    if [[ -d "${REPO_ROOT}/third-party/cutlass/include/cute" ]]; then
        CUTLASS_HOME="${REPO_ROOT}/third-party/cutlass"
    elif [[ -d "${REPO_ROOT}/third_party/cutlass/include/cute" ]]; then
        CUTLASS_HOME="${REPO_ROOT}/third_party/cutlass"
    elif [[ -d "${REPO_ROOT}/../cutlass/include/cute" ]]; then
        CUTLASS_HOME="$(cd "${REPO_ROOT}/../cutlass" && pwd)"
    else
        echo "Please set CUTLASS_HOME to a CUTLASS checkout containing include/cute"
        exit 1
    fi
fi

echo "[build_mega_ffn] CUDA_HOME   = ${CUDA_HOME}"
echo "[build_mega_ffn] CUTLASS_HOME= ${CUTLASS_HOME}"
echo "[build_mega_ffn] REPO_ROOT   = ${REPO_ROOT}"

OUT="${SCRIPT_DIR}/test_mega_ffn"
SRC="${SCRIPT_DIR}/test_sm100_fp8_mega_ffn.cu"

INCS=(
    "-I${REPO_ROOT}/deep_gemm/include"
    "-I${CUTLASS_HOME}/include"
    "-I${CUDA_HOME}/include"
)

# sm_100a: Blackwell B200 feature set (UMMA / TMEM / cluster-3 / ...)
FLAGS=(
    -std=c++17
    -O3
    -arch=sm_100a
    -Xcompiler=-fPIC
    -Xcompiler=-Wno-psabi
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -DCUTE_USE_PACKED_TUPLE=1
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
    # 避免编译器过度限制 register 使用
    --maxrregcount=255
)

LIBS=(
    "-L${CUDA_HOME}/lib64"
    -lcuda
    -lcudart
)

set -x
"${NVCC}" "${FLAGS[@]}" "${INCS[@]}" "${SRC}" "${LIBS[@]}" -o "${OUT}"
set +x

echo "[build_mega_ffn] built: ${OUT}"
