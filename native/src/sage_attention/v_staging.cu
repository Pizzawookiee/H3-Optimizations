// SPDX-License-Identifier: Apache-2.0
#include "dtype_dispatch.cuh"
#include "float_utils.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
constexpr int kDTile = 8;
__device__ __forceinline__ int inv_perm16_staging(int w) {
  return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) |
         (((w >> 2) & 1) << 3);
}

template <typename T>
__device__ __forceinline__ void load_tile_staging(const T *ptr, float *out_vals) {
  if constexpr (std::is_same_v<T, float>) {
    float4 v0 = *reinterpret_cast<const float4 *>(ptr);
    float4 v1 = *reinterpret_cast<const float4 *>(ptr + 4);
    out_vals[0] = v0.x; out_vals[1] = v0.y; out_vals[2] = v0.z; out_vals[3] = v0.w;
    out_vals[4] = v1.x; out_vals[5] = v1.y; out_vals[6] = v1.z; out_vals[7] = v1.w;
  } else {
    const T *loaded = comfy::load_f16x8(ptr);
#pragma unroll
    for (int i = 0; i < kDTile; ++i) out_vals[i] = static_cast<float>(loaded[i]);
  }
}

template <typename T, int THREADS>
__global__ void v_amax_chunk_kernel(const T *__restrict__ v, float *__restrict__ amax,
                                    int rows, int H, int D, int64_t sb, int64_t sh, int64_t sn) {
  const int d_tiles = D / kDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kDTile;
  const T *base = v + b * sb + h * sh + d0;
  constexpr int WARPS = THREADS / 32;
  float mx[kDTile];
#pragma unroll
  for (int i = 0; i < kDTile; ++i) mx[i] = 0.f;
  int n = threadIdx.x;
  const int body = rows - 3 * THREADS;
  for (; n < body; n += 4 * THREADS) {
    float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
    load_tile_staging(base + (int64_t)n * sn, t0);
    load_tile_staging(base + (int64_t)(n + THREADS) * sn, t1);
    load_tile_staging(base + (int64_t)(n + 2 * THREADS) * sn, t2);
    load_tile_staging(base + (int64_t)(n + 3 * THREADS) * sn, t3);
#pragma unroll
    for (int di = 0; di < kDTile; ++di) {
      float a = fabsf(t0[di]), bb = fabsf(t1[di]), c = fabsf(t2[di]), d = fabsf(t3[di]);
      mx[di] = fmaxf(mx[di], fmaxf(fmaxf(a, bb), fmaxf(c, d)));
    }
  }
  for (; n < rows; n += THREADS) {
    float tmp[kDTile];
    load_tile_staging(base + (int64_t)n * sn, tmp);
#pragma unroll
    for (int di = 0; di < kDTile; ++di) mx[di] = fmaxf(mx[di], fabsf(tmp[di]));
  }
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int di = 0; di < kDTile; ++di) mx[di] = comfy::warp_reduce_fmax(mx[di]);
  __shared__ float warp_mx[kDTile][WARPS];
  if (lane == 0) {
#pragma unroll
    for (int di = 0; di < kDTile; ++di) warp_mx[di][warp] = mx[di];
  }
  __syncthreads();
  if (threadIdx.x < kDTile) {
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) val = fmaxf(val, warp_mx[threadIdx.x][w]);
    const int64_t slot = (int64_t)(b * H + h) * D + d0 + threadIdx.x;
    amax[slot] = fmaxf(amax[slot], val);
  }
}

template <typename T, int THREADS>
__global__ void quant_v_chunk_into_kernel(const T *__restrict__ v, int8_t *__restrict__ out,
                                          const float *__restrict__ scale, int rows,
                                          int row_start, int padded_N, int H, int D,
                                          int64_t sb, int64_t sh, int64_t sn) {
  const int d_tiles = D / kDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kDTile;
  const T *base = v + b * sb + h * sh + d0;
  const int64_t out_row = (int64_t)(b * H + h) * D + d0;
  __shared__ float inv_sc_sh[kDTile];
  if (threadIdx.x < kDTile) inv_sc_sh[threadIdx.x] = 1.f / scale[out_row + threadIdx.x];
  __syncthreads();
  float inv_sc[kDTile];
#pragma unroll
  for (int di = 0; di < kDTile; ++di) inv_sc[di] = inv_sc_sh[di];
  for (int local = rows - 1 - (int)threadIdx.x; local >= 0; local -= THREADS) {
    const int src = row_start + local;
    const int dst = (src & ~15) | inv_perm16_staging(src & 15);
    float tmp[kDTile];
    load_tile_staging(base + (int64_t)local * sn, tmp);
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      out[(out_row + di) * padded_N + dst] = comfy::float_to_int8_rn(tmp[di] * inv_sc[di]);
  }
}

void check_v_layout(const void *v, int B, int H, int rows, int D, int64_t sb, int64_t sh,
                    int64_t sn, int input_dtype_code, const char *what) {
  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const auto stride_ok = [element_size](int64_t stride, int extent) {
    return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
  };
  if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(sb, B) ||
      !stride_ok(sh, H) || !stride_ok(sn, rows))
    throw std::runtime_error(std::string(what) + ": V pointer/strides must be 16-byte aligned");
  if (D <= 0 || D % kDTile != 0)
    throw std::runtime_error(std::string(what) + ": head_dim must be a positive multiple of 8");
  if (B <= 0 || H <= 0 || rows <= 0)
    throw std::runtime_error(std::string(what) + ": empty chunk");
}

void report_launch(const char *what) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string(what) + " kernel launch failed: " + cudaGetErrorString(error));
}
} // namespace

void launch_v_amax_chunk(const void *v, void *amax, int B, int H, int rows, int D,
                         int64_t sb, int64_t sh, int64_t sn, int input_dtype_code,
                         cudaStream_t stream) {
  check_v_layout(v, B, H, rows, D, sb, sh, sn, input_dtype_code, "v_amax_chunk");
  const int blocks = B * H * (D / kDTile);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (rows <= 256)
      v_amax_chunk_kernel<T, 128><<<blocks, 128, 0, stream>>>(static_cast<const T *>(v), static_cast<float *>(amax), rows, H, D, sb, sh, sn);
    else
      v_amax_chunk_kernel<T, 512><<<blocks, 512, 0, stream>>>(static_cast<const T *>(v), static_cast<float *>(amax), rows, H, D, sb, sh, sn);
  });
  report_launch("v_amax_chunk");
}

void launch_quant_v_chunk_into(const void *v, void *out, const void *scale, int B, int H,
                               int rows, int row_start, int D, int padded_N, int64_t sb,
                               int64_t sh, int64_t sn, int input_dtype_code,
                               cudaStream_t stream) {
  check_v_layout(v, B, H, rows, D, sb, sh, sn, input_dtype_code, "quantize_v_chunk_into");
  if (row_start < 0 || padded_N <= 0 || row_start + rows > padded_N)
    throw std::runtime_error("quantize_v_chunk_into: chunk outside padded sequence");
  if (scale == nullptr)
    throw std::runtime_error("quantize_v_chunk_into: finalized V scale required");
  const int blocks = B * H * (D / kDTile);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (rows <= 256)
      quant_v_chunk_into_kernel<T, 128><<<blocks, 128, 0, stream>>>(static_cast<const T *>(v), static_cast<int8_t *>(out), static_cast<const float *>(scale), rows, row_start, padded_N, H, D, sb, sh, sn);
    else
      quant_v_chunk_into_kernel<T, 512><<<blocks, 512, 0, stream>>>(static_cast<const T *>(v), static_cast<int8_t *>(out), static_cast<const float *>(scale), rows, row_start, padded_N, H, D, sb, sh, sn);
  });
  report_launch("quantize_v_chunk_into");
}
