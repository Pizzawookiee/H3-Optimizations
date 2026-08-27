// SPDX-License-Identifier: Apache-2.0

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int kDTile = 8;

__device__ __forceinline__ int inv_perm16_staging(int value) {
  return (value & 1) | (((value >> 3) & 1) << 1) |
         (((value >> 1) & 1) << 2) | (((value >> 2) & 1) << 3);
}

template <typename T>
__device__ __forceinline__ void load_tile_staging(const T *pointer,
                                                  float *values) {
  if constexpr (std::is_same_v<T, float>) {
    float4 first = *reinterpret_cast<const float4 *>(pointer);
    float4 second = *reinterpret_cast<const float4 *>(pointer + 4);
    values[0] = first.x;
    values[1] = first.y;
    values[2] = first.z;
    values[3] = first.w;
    values[4] = second.x;
    values[5] = second.y;
    values[6] = second.z;
    values[7] = second.w;
  } else {
    const T *loaded = comfy::load_f16x8(pointer);
#pragma unroll
    for (int index = 0; index < kDTile; ++index) {
      values[index] = static_cast<float>(loaded[index]);
    }
  }
}

template <typename T, int Threads>
__global__ void v_amax_chunk_kernel(const T *__restrict__ v,
                                    float *__restrict__ amax, int rows, int H,
                                    int D, int64_t sb, int64_t sh,
                                    int64_t sn) {
  const int d_tiles = D / kDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kDTile;
  const T *base = v + b * sb + h * sh + d0;
  constexpr int Warps = Threads / 32;

  float maximum[kDTile];
#pragma unroll
  for (int index = 0; index < kDTile; ++index) maximum[index] = 0.f;

  int row = threadIdx.x;
  const int body = rows - 3 * Threads;
  for (; row < body; row += 4 * Threads) {
    float first[kDTile], second[kDTile], third[kDTile], fourth[kDTile];
    load_tile_staging(base + (int64_t)row * sn, first);
    load_tile_staging(base + (int64_t)(row + Threads) * sn, second);
    load_tile_staging(base + (int64_t)(row + 2 * Threads) * sn, third);
    load_tile_staging(base + (int64_t)(row + 3 * Threads) * sn, fourth);
#pragma unroll
    for (int index = 0; index < kDTile; ++index) {
      maximum[index] = fmaxf(
          maximum[index],
          fmaxf(fmaxf(fabsf(first[index]), fabsf(second[index])),
                fmaxf(fabsf(third[index]), fabsf(fourth[index]))));
    }
  }
  for (; row < rows; row += Threads) {
    float values[kDTile];
    load_tile_staging(base + (int64_t)row * sn, values);
#pragma unroll
    for (int index = 0; index < kDTile; ++index) {
      maximum[index] = fmaxf(maximum[index], fabsf(values[index]));
    }
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int index = 0; index < kDTile; ++index) {
    maximum[index] = comfy::warp_reduce_fmax(maximum[index]);
  }

  __shared__ float warp_maximum[kDTile][Warps];
  if (lane == 0) {
#pragma unroll
    for (int index = 0; index < kDTile; ++index) {
      warp_maximum[index][warp] = maximum[index];
    }
  }
  __syncthreads();

  if (threadIdx.x < kDTile) {
    float value = 0.f;
#pragma unroll
    for (int warp_index = 0; warp_index < Warps; ++warp_index) {
      value = fmaxf(value, warp_maximum[threadIdx.x][warp_index]);
    }
    const int64_t slot = (int64_t)(b * H + h) * D + d0 + threadIdx.x;
    amax[slot] = fmaxf(amax[slot], value);
  }
}

template <typename T, int Threads>
__global__ void quant_v_chunk_into_kernel(
    const T *__restrict__ v, int8_t *__restrict__ out,
    const float *__restrict__ scale, int rows, int row_start, int padded_N,
    int H, int D, int64_t sb, int64_t sh, int64_t sn) {
  const int d_tiles = D / kDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kDTile;
  const T *base = v + b * sb + h * sh + d0;
  const int64_t out_row = (int64_t)(b * H + h) * D + d0;

  __shared__ float inverse_scale[kDTile];
  if (threadIdx.x < kDTile) {
    inverse_scale[threadIdx.x] = 1.f / scale[out_row + threadIdx.x];
  }
  __syncthreads();

  float inverse[kDTile];
#pragma unroll
  for (int index = 0; index < kDTile; ++index) {
    inverse[index] = inverse_scale[index];
  }

  for (int local = rows - 1 - (int)threadIdx.x; local >= 0;
       local -= Threads) {
    const int source = row_start + local;
    const int destination =
        (source & ~15) | inv_perm16_staging(source & 15);
    float values[kDTile];
    load_tile_staging(base + (int64_t)local * sn, values);
#pragma unroll
    for (int index = 0; index < kDTile; ++index) {
      out[(out_row + index) * padded_N + destination] =
          comfy::float_to_int8_rn(values[index] * inverse[index]);
    }
  }
}

void check_v_layout(const void *v, int B, int H, int rows, int D, int64_t sb,
                    int64_t sh, int64_t sn, int input_dtype_code,
                    const char *name) {
  const size_t element_size =
      input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const auto stride_ok = [element_size](int64_t stride, int extent) {
    return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
  };
  if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(sb, B) ||
      !stride_ok(sh, H) || !stride_ok(sn, rows)) {
    throw std::runtime_error(std::string(name) +
                             ": V pointer and B/H/N strides must be 16-byte aligned");
  }
  if (D <= 0 || D % kDTile != 0) {
    throw std::runtime_error(std::string(name) +
                             ": head_dim must be a positive multiple of 8");
  }
  if (B <= 0 || H <= 0 || rows <= 0) {
    throw std::runtime_error(std::string(name) + ": empty V chunk");
  }
}

void report_launch(const char *name) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(name) + " kernel launch failed: " +
                             cudaGetErrorString(error));
  }
}

} // namespace

void launch_v_amax_chunk(const void *v, void *amax, int B, int H, int rows,
                         int D, int64_t sb, int64_t sh, int64_t sn,
                         int input_dtype_code, cudaStream_t stream) {
  check_v_layout(v, B, H, rows, D, sb, sh, sn, input_dtype_code,
                 "v_amax_chunk");
  const int blocks = B * H * (D / kDTile);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (rows <= 256) {
      v_amax_chunk_kernel<T, 128><<<blocks, 128, 0, stream>>>(
          static_cast<const T *>(v), static_cast<float *>(amax), rows, H, D,
          sb, sh, sn);
    } else {
      v_amax_chunk_kernel<T, 512><<<blocks, 512, 0, stream>>>(
          static_cast<const T *>(v), static_cast<float *>(amax), rows, H, D,
          sb, sh, sn);
    }
  });
  report_launch("v_amax_chunk");
}

void launch_quant_v_chunk_into(const void *v, void *out, const void *scale,
                               int B, int H, int rows, int row_start, int D,
                               int padded_N, int64_t sb, int64_t sh,
                               int64_t sn, int input_dtype_code,
                               cudaStream_t stream) {
  check_v_layout(v, B, H, rows, D, sb, sh, sn, input_dtype_code,
                 "quantize_v_chunk_into");
  if (row_start < 0 || row_start + rows > padded_N || scale == nullptr) {
    throw std::runtime_error("quantize_v_chunk_into: invalid destination range");
  }
  const int blocks = B * H * (D / kDTile);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (rows <= 256) {
      quant_v_chunk_into_kernel<T, 128><<<blocks, 128, 0, stream>>>(
          static_cast<const T *>(v), static_cast<int8_t *>(out),
          static_cast<const float *>(scale), rows, row_start, padded_N, H, D,
          sb, sh, sn);
    } else {
      quant_v_chunk_into_kernel<T, 512><<<blocks, 512, 0, stream>>>(
          static_cast<const T *>(v), static_cast<int8_t *>(out),
          static_cast<const float *>(scale), rows, row_start, padded_N, H, D,
          sb, sh, sn);
    }
  });
  report_launch("quantize_v_chunk_into");
}
