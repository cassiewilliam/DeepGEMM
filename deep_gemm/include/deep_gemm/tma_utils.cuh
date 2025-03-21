#pragma once

#include <cassert>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

#include "utils.cuh"

namespace deep_gemm {

template <class T>
constexpr CUtensorMapDataType get_CUtensorMapDataType() {
    if constexpr (std::is_same<T, uint8_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if constexpr (std::is_same<T, __nv_fp8_e5m2>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    } else if constexpr (std::is_same<T, uint16_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else if constexpr (std::is_same<T, uint32_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    } else if constexpr (std::is_same<T, uint64_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    } else if constexpr (std::is_same<T, int32_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    } else if constexpr (std::is_same<T, int64_t>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_INT64;
    } else if constexpr (std::is_same<T, __half>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same<T, float>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }  else if constexpr (std::is_same<T, double>::value) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    }
}

// Get a function pointer to the cuTensorMapEncodeTiled driver API.
PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
    // Get pointer to `cuTensorMapEncodeTiled`
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;

#if CUDA_VERSION >= 12050
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                     cudaEnableDefault, &driver_status);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr,
                            cudaEnableDefault, &driver_status);
#endif

    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("driver_status != cudaDriverEntryPointSuccess");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(cuTensorMapEncodeTiled_ptr);
}

// 一维和多维情况的主要区别在于必须在主机上创建张量图并将其传递给 CUDA 内核
template <typename T>
CUtensorMap make_2d_tma_copy_desc(T* global_address, uint64_t gmem_dim[2],
                                  uint64_t stride_in_bytes, uint32_t smem_dim[2], // // The box_size is the size of the shared memory buffer that is used as the destination of a TMA transfer.
                                  CUtensorMapSwizzle swizzle_type,
                                  PFN_cuTensorMapEncodeTiled encode_func = nullptr) {
    CUtensorMap tensor_map{};

    // rank is the number of dimensions of the array.
    constexpr uint32_t rank = 2;

    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t global_stride[rank - 1] = {stride_in_bytes};

    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_strides[rank] = {1, 1};

    if (encode_func == nullptr)
        encode_func = get_cuTensorMapEncodeTiled();

    // Create the tensor descriptor.
    auto result = encode_func(
            &tensor_map, get_CUtensorMapDataType<typename std::remove_cv<T>::type>(), rank /*cuuint32_t tensorRank Tensor维度*/,
            global_address /*void *globalAddress*/, gmem_dim /*cuuint64_t *globalDim (in number of elements) */, 
            global_stride/*cuuint64_t *globalStrides 全局内存stride (in bytes)*/,
            smem_dim/*cuuint32_t *boxDim*/, elem_strides/*cuuint32_t *elementStrides*/,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
            swizzle_type,   // Swizzling can be used to avoid shared memory bank conflicts.
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,   // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);  // Any element that is outside of bounds will be set to zero by the TMA transfer. 越界处理
    DG_HOST_ASSERT(result == CUDA_SUCCESS);
    return tensor_map;
}

template <uint32_t kNumTMAMulticast = 1>
__device__ __forceinline__ void
tma_copy(void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr,
         int32_t const& crd_0, int32_t const& crd_1) {
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    if constexpr (kNumTMAMulticast == 1) {
        // "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
    } else if (cute::block_rank_in_cluster() == 0) {
        // multiple SMEM locations in multiple CTAs.
        // an input matrix column tile is needed for multiple row tiles or vice versa.
        // the .multicast operand allows us to guarantee L2-cache hits.
        // "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
        //  (1 << kNumTMAMulticast) - 1 表示ctaMask参数，用于标志当前数据需要广播到哪几个CTA中，当kNumTMAMulticast=2时，此时参数=3（011）,表示两个CTA都需要数据
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(desc_ptr, barrier_ptr, (1 << kNumTMAMulticast) - 1, cache_hint, smem_ptr, crd_0, crd_1);
    }
}

}  // namespace deep_gemm
