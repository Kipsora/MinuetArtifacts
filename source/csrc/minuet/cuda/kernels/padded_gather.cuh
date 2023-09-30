#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <std::size_t T_TILE_SIZE, typename IT, typename FT>
__global__ void PaddedGather(
    std::size_t num_sources,                            //
    std::size_t num_offsets,                            //
    std::size_t num_source_feature_tiles,               //
    const IT *__restrict__ cumsum_offset_padded_sizes,  //
    const IT *__restrict__ source_masks,                //
    const FT *__restrict__ sources,                     //
    FT *__restrict__ source_buffers) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);

  for (UIter i = gid; i < num_sources * num_source_feature_tiles; i += gsz) {
    auto s = i / num_source_feature_tiles;
    auto f = i % num_source_feature_tiles;

    auto source = sources + (s * num_source_feature_tiles + f) * T_TILE_SIZE;

    FT value[T_TILE_SIZE];
    device::Assign<T_TILE_SIZE>(value, source);
    for (UIter o = 0; o < num_offsets; o++) {
      auto d = source_masks[o * num_sources + s];
      if (d == -1) {
        continue;
      }
      d += cumsum_offset_padded_sizes[o];

      auto source_buffer =
          source_buffers + (d * num_source_feature_tiles + f) * T_TILE_SIZE;
      device::Assign<T_TILE_SIZE>(source_buffer, value);
    }
  }
}

}  // namespace minuet::cuda::kernels