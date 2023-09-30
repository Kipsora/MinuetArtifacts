#pragma once

#include "minuet/cuda/context.cuh"

namespace minuet::cuda::kernels {

template <std::size_t T_TILE_SIZE, typename IT, typename FT>
__global__ void PaddedScatter(
    std::size_t num_targets,                            //
    std::size_t num_offsets,                            //
    std::size_t num_target_feature_tiles,               //
    const IT *__restrict__ cumsum_offset_padded_sizes,  //
    const IT *__restrict__ target_masks,                //
    const FT *__restrict__ target_buffers,              //
    FT *__restrict__ targets) {
  auto gid = MINUET_GLOBAL_THREAD_ID(x);
  auto gsz = MINUET_N_GLOBAL_THREADS(x);
  for (UIter i = gid; i < num_targets * num_target_feature_tiles; i += gsz) {
    auto t = i / num_target_feature_tiles;
    auto f = i % num_target_feature_tiles;

    auto target = targets + (t * num_target_feature_tiles + f) * T_TILE_SIZE;

    FT value[T_TILE_SIZE];
    Iterate<UIter, T_TILE_SIZE>([&](UIter k) { value[k] = 0; });

    for (UIter o = 0; o < num_offsets; o++) {
      auto d = target_masks[o * num_targets + t];
      if (d == -1) {
        continue;
      }
      d += cumsum_offset_padded_sizes[o];
      auto target_buffer =
          target_buffers + (d * num_target_feature_tiles + f) * T_TILE_SIZE;
      Iterate<UIter, T_TILE_SIZE>(
          [&](auto k) { value[k] += target_buffer[k]; });
    }
    device::Assign<T_TILE_SIZE>(target, value);
  }
}

}  // namespace minuet::cuda::kernels
