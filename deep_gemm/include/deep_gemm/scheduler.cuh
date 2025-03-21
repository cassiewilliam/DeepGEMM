#include "utils.cuh"

namespace deep_gemm {

enum class GemmType {
    Normal,
    GroupedContiguous,
    GroupedMasked
};

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t kNumGroups, uint32_t kNumTMAMulticast,
          uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>   // kNumNBlocksPerGroup = 16, 控制SWIZZLE, 16表示每个Group沿着N方向计算16个N-Blocks
struct Scheduler {
    int current_iter = -1;
    uint32_t num_aligned_m_blocks;

    // For normal GEMM
    // Maybe not used in the masked grouped GEMM
    uint32_t num_blocks;

    // For grouped GEMM
    int* grouped_layout;
    // Only used for masked layout
    uint32_t curr_group_idx, curr_cumsum;

    __device__ __forceinline__ explicit Scheduler(const uint32_t shape_m,
                                                  int* grouped_layout = nullptr) {
        num_aligned_m_blocks = ceil_div(shape_m, BLOCK_M);
        if constexpr (kGemmType == GemmType::Normal) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            num_blocks = num_aligned_m_blocks * kNumNBlocks;
            this->grouped_layout = grouped_layout;
        } else if (kGemmType == GemmType::GroupedMasked) {
            curr_group_idx = curr_cumsum = 0;
            this->grouped_layout = grouped_layout;
        }
    }

    __device__ __forceinline__ void get_swizzled_block_idx(const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx) {
        DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

        // 注意：这里的Group理解为一个CTAs SWIZZLE块，不是warp group!
        // kNumNBlocksPerGroup = 16; kNumNBlocks=128;
        // num_m_blocks m维度切分block总数  256/128 = 2
        // block_idx 当前该计算的 block_idx [0, 132, 264 ...]
        // Swizzle for better L2 usages

        // block_idx = 0/128
        auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup;  // 每个wave能够计算的block总数; 2*16 = 32
        auto group_idx = block_idx / num_blocks_per_group;          // 0/32=0; 128/32=4; 当前block_idx该计算哪个group了
        auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;   // 0*16=0;  4*16 = 64; 前面已经计算完多少个完整的Block块
        auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);  // min(16, 128) = 16; min(16, 64) = 16; 在这个group内，还剩余多少N块该计算
        auto in_group_idx = block_idx % num_blocks_per_group;   // 0; 128%32=0; 在当前group内，该计算哪一块了
        m_block_idx = in_group_idx / num_n_blocks_in_group;     // 0; idx/n_blcoks = m_blocks_idx
        n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;  // 64;  求n_blocks_idx
    }

    template <bool kIgnoreGroupedForGroupedContiguous=true>
    __device__ __forceinline__ uint32_t get_global_idx(const uint32_t shape_dim, const uint32_t block_size,
                                                       const uint32_t& block_idx, const uint32_t& m_block_idx=0) {
        if constexpr (kGemmType == GemmType::Normal) {
            return block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedContiguous) {
            // Contiguous Layout中，通过grouped_layout+m_idx获取当前 m块 对应哪个 expert，从而获取到正确的expert坐标
            auto offset = kIgnoreGroupedForGroupedContiguous ? 0 : __ldg(grouped_layout + m_block_idx * BLOCK_M);
            return offset * shape_dim + block_idx * block_size;
        } else if (kGemmType == GemmType::GroupedMasked) {
            return curr_group_idx * shape_dim + block_idx * block_size;
        }
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
        const auto next_block_idx = (++ current_iter) * gridDim.x + blockIdx.x;

        if constexpr (kGemmType == GemmType::GroupedMasked) {
            uint32_t num_m_blocks;
            while (true) {
                // End of the task
                if (curr_group_idx == kNumGroups)
                    return false;

                // Within current group
                // 针对masked layout，一个一个gemm计算，获取当前gemm真实的m维度，计算num_m_blocks
                num_m_blocks = ceil_div(static_cast<uint32_t>(__ldg(grouped_layout + curr_group_idx)), BLOCK_M);
                auto current_m_block_cumsum = curr_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * kNumNBlocks)  //还在当前group内
                    break;

                // Move to check the next group
                curr_group_idx ++, curr_cumsum = current_m_block_cumsum;
            }

            get_swizzled_block_idx(num_m_blocks, next_block_idx - curr_cumsum * kNumNBlocks, m_block_idx, n_block_idx);
        } else {
            if (next_block_idx >= num_blocks)
                return false;

            get_swizzled_block_idx(num_aligned_m_blocks, next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};
#pragma clang diagnostic pop

} // namespace deep_gemm
