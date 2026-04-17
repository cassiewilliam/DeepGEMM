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
OUT_BASELINE="${SCRIPT_DIR}/test_mega_ffn_baseline"
SRC_BASELINE="${SCRIPT_DIR}/test_mega_ffn_baseline.cu"

INCS=(
    "-I${REPO_ROOT}/deep_gemm/include"
    "-I${CUTLASS_HOME}/include"
    "-I${CUDA_HOME}/include"
)

# sm_100a: Blackwell B200 feature set (UMMA / TMEM / cluster-3 / ...)
# 注意：CUDA 13 必须用 -gencode 指定 arch-specific 后缀 100a，否则 ptxas
# 会以 sm_100 target 解析，导致 tcgen05.* / cta_group::1 / mxf8f6f4 等指令报错。
FLAGS=(
    -std=c++17
    -O3
    -gencode=arch=compute_100a,code=sm_100a
    -Xcompiler=-fPIC
    -Xcompiler=-Wno-psabi
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -DCUTE_USE_PACKED_TUPLE=1
    -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
)

# 允许通过 TRACE=1 ./build_mega_ffn.sh 打开 kernel 内 printf 调试
if [[ "${TRACE:-0}" == "1" ]]; then
    FLAGS+=(-DMEGA_FFN_TRACE=1)
fi

# 允许环境变量注入 kernel 模板参数 (方便 perf 扫描)
#   STAGES=4 EPI_THREADS=256 CLUSTER_DIM=2 ./build_mega_ffn.sh
if [[ -n "${STAGES:-}" ]];      then FLAGS+=(-DMFFN_STAGES=${STAGES}); fi
if [[ -n "${EPI_THREADS:-}" ]]; then FLAGS+=(-DMFFN_EPI_THREADS=${EPI_THREADS}); fi
if [[ -n "${CLUSTER_DIM:-}" ]]; then FLAGS+=(-DMFFN_CLUSTER_DIM=${CLUSTER_DIM}); fi
if [[ -n "${L2_K_SPLIT:-}" ]];  then FLAGS+=(-DMFFN_L2_K_SPLIT=${L2_K_SPLIT}); fi

LIBS=(
    "-L${CUDA_HOME}/lib64"
    -lcuda
    -lcudart
)

LIBS_BASELINE=(
    "-L${CUDA_HOME}/lib64"
    -lcuda
    -lcudart
    -lcublas
)

set -x
"${NVCC}" "${FLAGS[@]}" "${INCS[@]}" "${SRC}" "${LIBS[@]}" -o "${OUT}"
set +x
echo "[build_mega_ffn] built: ${OUT}"

# baseline (optional; 设 SKIP_BASELINE=1 跳过)
if [[ "${SKIP_BASELINE:-0}" != "1" ]] && [[ -f "${SRC_BASELINE}" ]]; then
    set -x
    "${NVCC}" "${FLAGS[@]}" "${INCS[@]}" "${SRC_BASELINE}" "${LIBS_BASELINE[@]}" -o "${OUT_BASELINE}"
    set +x
    echo "[build_mega_ffn] built: ${OUT_BASELINE}"
fi
