// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Optimized INT8 per-thread quantization for Q and K (SageAttention).
//
// All tensors assumed contiguous [B, H, L, D] (HND layout).
// Two-kernel launch: Q and K in separate kernels for better I-cache
// utilization, with a fused fallback for non-standard tile configs.
//   Q path: warp-per-group, single-pass, vectorized float2 loads / int32
//   stores. K path: warp-per-group, single-pass, vectorized loads / stores.
//   No __syncthreads anywhere – pure warp-level reductions.
//
// Block / warp tile sizes and alignment are template parameters so the
// compiler can constant-fold address arithmetic (divisions, modulos) and
// eliminate dead scalar-fallback code when C is a multiple of 4.
//
#include "dtype_dispatch.cuh"
#include "float_utils.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

using comfy::quant_int8_rcp;
using comfy::store4_i8;
using comfy::warp_reduce_fmax;

namespace {

template <typename T>
struct VectorLoader4;

template <>
struct VectorLoader4<half> {
  __forceinline__ __device__ static void load(const half* ptr, float* out) {
    float2 raw = __ldg(reinterpret_cast<const float2 *>(ptr));
    const half *vals = reinterpret_cast<const half *>(&raw);
    out[0] = static_cast<float>(vals[0]);
    out[1] = static_cast<float>(vals[1]);
    out[2] = static_cast<float>(vals[2]);
    out[3] = static_cast<float>(vals[3]);
  }
};

template <>
struct VectorLoader4<nv_bfloat16> {
  __forceinline__ __device__ static void load(const nv_bfloat16* ptr, float* out) {
    float2 raw = __ldg(reinterpret_cast<const float2 *>(ptr));
    const nv_bfloat16 *vals = reinterpret_cast<const nv_bfloat16 *>(&raw);
    out[0] = static_cast<float>(vals[0]);
    out[1] = static_cast<float>(vals[1]);
    out[2] = static_cast<float>(vals[2]);
    out[3] = static_cast<float>(vals[3]);
  }
};

template <>
struct VectorLoader4<float> {
  __forceinline__ __device__ static void load(const float* ptr, float* out) {
    float4 raw = __ldg(reinterpret_cast<const float4 *>(ptr));
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
  }
};

__forceinline__ __device__ void convrot4(float *values) {
  const float x0 = values[0];
  const float x1 = values[1];
  const float x2 = values[2];
  const float x3 = values[3];
  const float a0 = x0 + x1;
  const float a1 = x0 - x1;
  const float a2 = x2 + x3;
  const float a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}

// A fixed random diagonal makes the following Hadamard a randomized
// orthogonal transform instead of aligning every row to the same structured
// basis. Q and K use the same signs, so their exact dot product is unchanged.
// Flip the IEEE sign bit directly: this is exact and avoids an FP multiply.
__forceinline__ __device__ void apply_convrot_sign128(float *values,
                                                      const int lane) {
  constexpr uint32_t signs_0 = 0x1035997bu;
  constexpr uint32_t signs_1 = 0x8087f5eeu;
  constexpr uint32_t signs_2 = 0xee2e4e1au;
  constexpr uint32_t signs_3 = 0x71132418u;
  const uint32_t signs =
      lane < 8    ? signs_0
      : lane < 16 ? signs_1
      : lane < 24 ? signs_2
                  : signs_3;
  const int shift = (lane & 7) * 4;
#pragma unroll
  for (int channel = 0; channel < 4; ++channel) {
    const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
    values[channel] =
        __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
  }
}

// Apply a normalized Walsh-Hadamard H64 to one 64-channel half-warp group.
// H4 covers the four adjacent channels owned by each lane; four shuffle
// butterflies cover the remaining 16-lane dimension. This uses half as many
// shuffles as evaluating H4 x H4 x H4 directly.
__forceinline__ __device__ void convrot64(float *values) {
  convrot4(values);
  const int half_lane = threadIdx.x & 15;
  const unsigned mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;

#pragma unroll
  for (int bit = 1; bit < 16; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(mask, values[c], bit, 16);
      values[c] =
          (half_lane & bit) ? other - values[c] : values[c] + other;
    }
  }

#pragma unroll
  for (int c = 0; c < 4; ++c)
    values[c] *= 0.25f;
}

// Apply a normalized Walsh-Hadamard H128 across all four-channel warp groups.
__forceinline__ __device__ void convrot128_plain(float *values) {
  convrot4(values);
  const int lane = threadIdx.x & 31;

#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float other = __shfl_xor_sync(0xffffffffu, values[c], bit);
      values[c] = (lane & bit) ? other - values[c] : values[c] + other;
    }
  }

#pragma unroll
  for (int c = 0; c < 4; ++c)
    values[c] *= 0.1767766952966369f;
}

__forceinline__ __device__ void convrot128(float *values) {
  apply_convrot_sign128(values, threadIdx.x & 31);
  convrot128_plain(values);
}

// ---------------------------------------------------------------------------
// Q processing device function
// ---------------------------------------------------------------------------
#pragma nv_diag_suppress 1056
template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__forceinline__ __device__ void
process_q(const T *__restrict__ in, int8_t *__restrict__ out,
          float *__restrict__ sc_buf, const int oblk, const int L,
          const int C, const int64_t stride_n) {
  constexpr int NSUB = BLKQ / WARPQ;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;

#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int otld = wid * 2 + g;
    const int base = (oblk / NSUB) * BLKQ + (oblk % NSUB) * WARPQ + otld;

    float v[CHANNEL_TILES * NR * 4];
    float mx = 0.f;

#pragma unroll
    for (int i = 0; i < CHANNEL_TILES * NR * 4; ++i)
      v[i] = 0.f;

#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int tile_base = tile * NR * 4;
      if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
            VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                                 fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
          }
        }
      } else if (ch < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              v[vi + c] =
                  (ch + c < C)
                      ? static_cast<float>(
                            __ldg(&in[(int64_t)n * stride_n + ch + c]))
                      : 0.f;
              mx = fmaxf(mx, fabsf(v[vi + c]));
            }
          }
        }
      }
    }

    if constexpr (ROTATION != 0) {
#pragma unroll
      for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          if constexpr (ROTATION == 128)
            convrot128(&v[(tile * NR + j) * 4]);
          else if constexpr (ROTATION == 129)
            convrot128_plain(&v[(tile * NR + j) * 4]);
          else if constexpr (ROTATION == 64)
            convrot64(&v[(tile * NR + j) * 4]);
          else
            convrot4(&v[(tile * NR + j) * 4]);
        }
      }
      mx = 0.f;
#pragma unroll
      for (int j = 0; j < CHANNEL_TILES * NR * 4; ++j)
        mx = fmaxf(mx, fabsf(v[j]));
    }

    mx = warp_reduce_fmax(mx);
    const float sc = mx / 127.f + 1e-7f;
    const float inv_sc = 1.f / sc;

    if (lane == 0)
      sc_buf[oblk * 8 + otld] = sc;

#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const int tile_base = tile * NR * 4;
      if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
            store4_i8(&out[(int64_t)n * C + ch],
                      quant_int8_rcp(v[vi], inv_sc),
                      quant_int8_rcp(v[vi + 1], inv_sc),
                      quant_int8_rcp(v[vi + 2], inv_sc),
                      quant_int8_rcp(v[vi + 3], inv_sc));
          }
        }
      } else if (ch < C) {
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const int n = base + j * 8;
          const int vi = tile_base + j * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (ch + c < C)
                out[(int64_t)n * C + ch + c] =
                    quant_int8_rcp(v[vi + c], inv_sc);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// K processing device function
//
// When anchor_index is nonnegative, subtracts that key vector from every key.
// The shift is exactly softmax-invariant and happens before abs-max reduction,
// rotation, and quantization.
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int CHANNEL_TILES, int ROTATION,
          bool ALIGNED4>
__forceinline__ __device__ void
process_k(const T *__restrict__ in, int8_t *__restrict__ out,
          float *__restrict__ sc_buf, const int oblk, const int L, const int C,
          const int anchor_index, const int64_t stride_n,
          const T *__restrict__ anchor_values) {
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int otld = wid;

  float bias[CHANNEL_TILES * 4];
#pragma unroll
  for (int i = 0; i < CHANNEL_TILES * 4; ++i)
    bias[i] = 0.f;

  if (anchor_index >= 0) {
#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
      const int ch = tile * 128 + (lane << 2);
      const T *anchor = anchor_values != nullptr
                            ? anchor_values + ch
                            : in + (int64_t)anchor_index * stride_n + ch;
      if (ALIGNED4 || ch + 3 < C) {
        VectorLoader4<T>::load(anchor, &bias[tile * 4]);
      } else if (ch < C) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          bias[tile * 4 + c] =
              (ch + c < C)
                  ? static_cast<float>(__ldg(&anchor[c]))
                  : 0.f;
      }
    }
  }

  float v[CHANNEL_TILES * 2 * NL * 4];
  float mx = 0.f;

#pragma unroll
  for (int i = 0; i < CHANNEL_TILES * 2 * NL * 4; ++i)
    v[i] = 0.f;

#pragma unroll
  for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
    const int ch = tile * 128 + (lane << 2);
    const int tile_base = tile * 2 * NL * 4;
    if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
            VectorLoader4<T>::load(&in[(int64_t)n * stride_n + ch], &v[vi]);
#pragma unroll
            for (int c = 0; c < 4; ++c)
              v[vi + c] -= bias[tile * 4 + c];
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v[vi]), fabsf(v[vi + 1])),
                                 fmaxf(fabsf(v[vi + 2]), fabsf(v[vi + 3]))));
          }
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              v[vi + c] =
                  (ch + c < C)
                      ? static_cast<float>(
                            __ldg(&in[(int64_t)n * stride_n + ch + c])) -
                            bias[tile * 4 + c]
                      : 0.f;
              mx = fmaxf(mx, fabsf(v[vi + c]));
            }
          }
        }
      }
    }
  }

  if constexpr (ROTATION != 0) {
#pragma unroll
    for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
#pragma unroll
      for (int j = 0; j < 2 * NL; ++j) {
        if constexpr (ROTATION == 128)
          convrot128(&v[(tile * 2 * NL + j) * 4]);
        else if constexpr (ROTATION == 129)
          convrot128_plain(&v[(tile * 2 * NL + j) * 4]);
        else if constexpr (ROTATION == 64)
          convrot64(&v[(tile * 2 * NL + j) * 4]);
        else
          convrot4(&v[(tile * 2 * NL + j) * 4]);
      }
    }
    mx = 0.f;
#pragma unroll
    for (int j = 0; j < CHANNEL_TILES * 2 * NL * 4; ++j)
      mx = fmaxf(mx, fabsf(v[j]));
  }

  mx = warp_reduce_fmax(mx);
  const float sc = mx / 127.f + 1e-7f;
  const float inv_sc = 1.f / sc;

  if (lane == 0)
    sc_buf[oblk * 4 + otld] = sc;

#pragma unroll
  for (int tile = 0; tile < CHANNEL_TILES; ++tile) {
    const int ch = tile * 128 + (lane << 2);
    const int tile_base = tile * 2 * NL * 4;
    if (ALIGNED4 || ch + 3 < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
            store4_i8(&out[(int64_t)n * C + ch],
                      quant_int8_rcp(v[vi], inv_sc),
                      quant_int8_rcp(v[vi + 1], inv_sc),
                      quant_int8_rcp(v[vi + 2], inv_sc),
                      quant_int8_rcp(v[vi + 3], inv_sc));
          }
        }
      }
    } else if (ch < C) {
#pragma unroll
      for (int j = 0; j < NL; ++j) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          const int n = oblk * WARPK + j * 8 + otld * 2 + p;
          const int vi = tile_base + (j * 2 + p) * 4;
          if (n < L) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              if (ch + c < C)
                out[(int64_t)n * C + ch + c] =
                    quant_int8_rcp(v[vi + c], inv_sc);
            }
          }
        }
      }
    }
  }
}
#pragma nv_diag_default 1056

// ---------------------------------------------------------------------------
// Model-independent K stabilization detector
//
// Samples nine evenly spaced keys per (batch, head), selects the sampled key
// that minimizes residual energy, and enables centering only when that anchor
// reduces sampled energy without increasing sampled abs-max by more than
// 12.5%. The output is an absolute sequence index, or -1 when the original K
// range is preferable. No host synchronization is required.
// ---------------------------------------------------------------------------
constexpr int CENTER_DETECT_THREADS = 128;
constexpr int CENTER_SAMPLES = 9;
constexpr int CENTER_MAX_CHANNELS = 256;

template <typename T>
__global__ __launch_bounds__(CENTER_DETECT_THREADS) void detect_k_anchor(
    const T *__restrict__ k_in, int *__restrict__ anchor_indices, const int Lk,
    const int C, const int H_kv, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n,
    const int *__restrict__ sample_positions,
    T *__restrict__ anchor_values) {
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int64_t bh_offset =
      (int64_t)b * stride_b + (int64_t)h * stride_h;

  __shared__ float samples[CENTER_SAMPLES * CENTER_MAX_CHANNELS];
  __shared__ float warp_original_energy[4];
  __shared__ float warp_original_max[4];
  __shared__ float warp_candidate_distance[CENTER_SAMPLES][4];
  __shared__ float warp_best_energy[4];
  __shared__ float warp_best_max[4];
  __shared__ int selected_candidate;
  __shared__ bool use_anchor;

  for (int index = tid; index < CENTER_SAMPLES * C;
       index += CENTER_DETECT_THREADS) {
    const int sample = index / C;
    const int channel = index - sample * C;
    const int row = sample_positions == nullptr
                        ? sample * (Lk - 1) / (CENTER_SAMPLES - 1)
                        : sample;
    samples[index] = static_cast<float>(
        __ldg(&k_in[bh_offset + (int64_t)row * stride_n + channel]));
  }
  __syncthreads();

  float original_energy = 0.f;
  float original_max = 0.f;
  float candidate_distance[CENTER_SAMPLES];
#pragma unroll
  for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
    candidate_distance[candidate] = 0.f;
  }

  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    float channel_sum = 0.f;
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float value = samples[sample * C + channel];
      original_energy = fmaf(value, value, original_energy);
      original_max = fmaxf(original_max, fabsf(value));
      channel_sum += value;
    }
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      const float distance =
          CENTER_SAMPLES * samples[candidate * C + channel] - channel_sum;
      candidate_distance[candidate] =
          fmaf(distance, distance, candidate_distance[candidate]);
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    original_energy +=
        __shfl_down_sync(0xffffffffu, original_energy, offset);
    original_max = fmaxf(
        original_max, __shfl_down_sync(0xffffffffu, original_max, offset));
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      candidate_distance[candidate] += __shfl_down_sync(
          0xffffffffu, candidate_distance[candidate], offset);
    }
  }

  if (lane == 0) {
    warp_original_energy[warp] = original_energy;
    warp_original_max[warp] = original_max;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      warp_candidate_distance[candidate][warp] =
          candidate_distance[candidate];
    }
  }
  __syncthreads();

  if (tid == 0) {
    int best_candidate = 0;
    float best_distance = 3.402823466e+38F;
#pragma unroll
    for (int candidate = 0; candidate < CENTER_SAMPLES; ++candidate) {
      float distance = 0.f;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        distance += warp_candidate_distance[candidate][w];
      }
      if (distance < best_distance) {
        best_candidate = candidate;
        best_distance = distance;
      }
    }
    selected_candidate = best_candidate;
  }
  __syncthreads();

  float best_energy = 0.f;
  float best_max = 0.f;
  for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
    const float anchor = samples[selected_candidate * C + channel];
#pragma unroll
    for (int sample = 0; sample < CENTER_SAMPLES; ++sample) {
      const float residual = samples[sample * C + channel] - anchor;
      best_energy = fmaf(residual, residual, best_energy);
      best_max = fmaxf(best_max, fabsf(residual));
    }
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    best_energy += __shfl_down_sync(0xffffffffu, best_energy, offset);
    best_max = fmaxf(
        best_max, __shfl_down_sync(0xffffffffu, best_max, offset));
  }
  if (lane == 0) {
    warp_best_energy[warp] = best_energy;
    warp_best_max[warp] = best_max;
  }
  __syncthreads();

  if (tid == 0) {
    float total_original_energy = 0.f;
    float total_original_max = 0.f;
    float total_best_energy = 0.f;
    float total_best_max = 0.f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      total_original_energy += warp_original_energy[w];
      total_original_max =
          fmaxf(total_original_max, warp_original_max[w]);
      total_best_energy += warp_best_energy[w];
      total_best_max = fmaxf(total_best_max, warp_best_max[w]);
    }

    use_anchor =
        total_best_energy < total_original_energy &&
        total_best_max <= total_original_max * 1.125f;
    anchor_indices[b * H_kv + h] =
        use_anchor
            ? (sample_positions == nullptr
                   ? selected_candidate * (Lk - 1) / (CENTER_SAMPLES - 1)
                   : sample_positions[selected_candidate])
            : -1;
  }
  __syncthreads();

  if (anchor_values != nullptr) {
    const int64_t anchor_offset = ((int64_t)b * H_kv + h) * C;
    for (int channel = tid; channel < C; channel += CENTER_DETECT_THREADS) {
      anchor_values[anchor_offset + channel] =
          use_anchor
              ? static_cast<T>(samples[selected_candidate * C + channel])
              : static_cast<T>(0.f);
    }
  }
}

// ---------------------------------------------------------------------------
// Standalone Q kernel
// ---------------------------------------------------------------------------
template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_q_kernel(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const int Lq, const int C, const int H_q,
    const int q_sc_per_h, const int64_t stride_b, const int64_t stride_h,
    const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
  const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
  process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
      q_in + in_bh, q_out + out_bh, q_sb + sbh, oblk, Lq, C, stride_n);
}

// ---------------------------------------------------------------------------
// Standalone K kernel
// ---------------------------------------------------------------------------
template <typename T, int NL, int WARPK, int CHANNEL_TILES, int ROTATION,
          bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_k_kernel(
    const T *__restrict__ k_in, int8_t *__restrict__ k_out,
    float *__restrict__ k_sb, const int *__restrict__ anchor_indices,
    const int Lk, const int C,
    const int H_kv, const int k_sc_per_h, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
  const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
  const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
  process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
      k_in + in_bh, k_out + out_bh, k_sb + sbh, oblk, Lk, C, anchor_index,
      stride_n, nullptr);
}

// ---------------------------------------------------------------------------
// Fused Q+K kernel – fallback for non-standard tile configs.
// blockIdx.x < q_oblk_count  →  Q path
// blockIdx.x >= q_oblk_count →  K path
// ---------------------------------------------------------------------------
template <typename T, int NR, int NL, int BLKQ, int WARPQ, int BLKK, int WARPK,
          int CHANNEL_TILES, int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 3) void quant_qk_fused(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const T *__restrict__ k_in,
    int8_t *__restrict__ k_out, float *__restrict__ k_sb,
    const int *__restrict__ anchor_indices,
    const int Lq, const int Lk, const int C, const int q_oblk_count,
    const int H_q, const int H_kv, const int q_sc_per_h, const int k_sc_per_h,
    const int64_t q_stride_b, const int64_t q_stride_h,
    const int64_t q_stride_n, const int64_t k_stride_b,
    const int64_t k_stride_h, const int64_t k_stride_n) {
  const int h = blockIdx.y, b = blockIdx.z;

  if (blockIdx.x < (unsigned)q_oblk_count) {
    if (h >= H_q)
      return;
    const int64_t in_bh = (int64_t)b * q_stride_b + (int64_t)h * q_stride_h;
    const int64_t out_bh = ((int64_t)b * H_q + h) * Lq * C;
    const int64_t sbh = ((int64_t)b * H_q + h) * q_sc_per_h;
    process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
        q_in + in_bh, q_out + out_bh, q_sb + sbh, blockIdx.x, Lq, C,
        q_stride_n);
  } else {
    if (h >= H_kv)
      return;
    const int64_t in_bh = (int64_t)b * k_stride_b + (int64_t)h * k_stride_h;
    const int64_t out_bh = ((int64_t)b * H_kv + h) * Lk * C;
    const int64_t sbh = ((int64_t)b * H_kv + h) * k_sc_per_h;
    const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
    process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
        k_in + in_bh, k_out + out_bh, k_sb + sbh,
        (int)blockIdx.x - q_oblk_count, Lk, C, anchor_index, k_stride_n,
        nullptr);
  }
}

template <typename T, int NR, int BLKQ, int WARPQ, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_q_into_kernel(
    const T *__restrict__ q_in, int8_t *__restrict__ q_out,
    float *__restrict__ q_sb, const int Lq, const int full_Lq,
    const int q_start, const int C, const int H_q, const int q_sc_per_h,
    const int64_t stride_b, const int64_t stride_h,
    const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh =
      ((int64_t)b * H_q + h) * full_Lq * C + (int64_t)q_start * C;
  const int q_scale_start =
      (q_start / BLKQ) * (BLKQ / WARPQ) * 8;
  const int64_t sbh =
      ((int64_t)b * H_q + h) * q_sc_per_h + q_scale_start;
  process_q<T, NR, BLKQ, WARPQ, CHANNEL_TILES, ROTATION, ALIGNED4>(
      q_in + in_bh, q_out + out_bh, q_sb + sbh, oblk, Lq, C, stride_n);
}

template <typename T, int NL, int BLKK, int WARPK, int CHANNEL_TILES,
          int ROTATION, bool ALIGNED4>
__global__ __launch_bounds__(128, 4) void quant_k_into_kernel(
    const T *__restrict__ k_in, int8_t *__restrict__ k_out,
    float *__restrict__ k_sb, const T *__restrict__ anchor_values,
    const int *__restrict__ anchor_indices, const int Lk,
    const int full_Lk, const int k_start, const int C, const int H_kv,
    const int k_sc_per_h, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_n) {
  const int oblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int64_t in_bh = (int64_t)b * stride_b + (int64_t)h * stride_h;
  const int64_t out_bh =
      ((int64_t)b * H_kv + h) * full_Lk * C + (int64_t)k_start * C;
  const int k_scale_start =
      (k_start / BLKK) * (BLKK / WARPK) * 4;
  const int64_t sbh =
      ((int64_t)b * H_kv + h) * k_sc_per_h + k_scale_start;
  const int anchor_index = __ldg(&anchor_indices[b * H_kv + h]);
  const T *anchor = anchor_values + ((int64_t)b * H_kv + h) * C;
  process_k<T, NL, WARPK, CHANNEL_TILES, ROTATION, ALIGNED4>(
      k_in + in_bh, k_out + out_bh, k_sb + sbh, oblk, Lk, C,
      anchor_index, stride_n, anchor);
}

} // namespace

void launch_quant_qk_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ,
    int WARPQ, int BLKK, int WARPK, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, void *anchor_indices,
    cudaStream_t stream) {
  if (C != 64 && C != 128 && C != 256) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: padded head_dim must be 64, 128, or 256");
  }
  const int expected_warpq = C == 256 ? 16 : 32;
  if (BLKQ != 128 || WARPQ != expected_warpq ||
      (BLKK != 64 && BLKK != 128) || WARPK != BLKK ||
      (C == 64 && BLKK != 64)) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: unsupported block/warp configuration");
  }
  if (!anchor_indices) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: anchor_indices scratch is required");
  }
  // An extent of one is exempt: its index is always zero, so the stride never
  // reaches an address, and PyTorch rewrites such a stride to 1 on the way
  // through DLPack (ATen/DLConvertor.cpp, gh-83069) on 2.9 and older, which no
  // caller can prevent.
  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const size_t vector_size = 4 * element_size;
  const auto stride_ok = [](int64_t stride, int extent) {
    return extent < 2 || (stride > 0 && stride % 4 == 0);
  };
  const auto is_vector_aligned = [vector_size, &stride_ok](const void *ptr, int64_t stride_b,
                                                            int64_t stride_h, int64_t stride_n,
                                                            int extent_b, int extent_h,
                                                            int extent_n) {
    return reinterpret_cast<uintptr_t>(ptr) % vector_size == 0 &&
           stride_ok(stride_b, extent_b) && stride_ok(stride_h, extent_h) &&
           stride_ok(stride_n, extent_n);
  };
  if (!is_vector_aligned(q, q_stride_b, q_stride_h, q_stride_n, B, H_q, Lq) ||
      !is_vector_aligned(k, k_stride_b, k_stride_h, k_stride_n, B, H_kv, Lk)) {
    throw std::runtime_error(
        "quant_qk_per_thread_int8: Q/K base pointers and B/H/N strides must preserve 4-element alignment");
  }

  const int q_oblk = (Lq + BLKQ - 1) / BLKQ * (BLKQ / WARPQ);
  const int k_oblk = (Lk + BLKK - 1) / BLKK * (BLKK / WARPK);
  const int q_sc_per_h = q_oblk * 8;
  const int k_sc_per_h = k_oblk * 4;
  auto *anchor_ptr = static_cast<int *>(anchor_indices);
  dim3 gd(H_kv, B);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    detect_k_anchor<T><<<gd, CENTER_DETECT_THREADS, 0, stream>>>(
        (const T *)k, anchor_ptr, Lk, C, H_kv, k_stride_b, k_stride_h,
        k_stride_n, nullptr, nullptr);
  });
  cudaError_t anchor_error = cudaGetLastError();
  if (anchor_error != cudaSuccess) {
    throw std::runtime_error(std::string("detect_k_anchor kernel launch failed: ") +
                             cudaGetErrorString(anchor_error));
  }

#define LAUNCH_SPLIT(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    dim3 gq(q_oblk, H_q, B);                                                   \
    quant_q_kernel<T, NR, BQ, WQ, CT, ROT, A4>                                 \
        <<<gq, 128, 0, stream>>>((const T *)q, (int8_t *)q_int8,               \
                                 (float *)q_scale, Lq, C, H_q, q_sc_per_h,     \
                                 q_stride_b, q_stride_h, q_stride_n);           \
    cudaError_t q_error = cudaGetLastError();                                   \
    if (q_error != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("quant_q kernel launch failed: ") + \
                               cudaGetErrorString(q_error));                    \
    dim3 gk(k_oblk, H_kv, B);                                                  \
    quant_k_kernel<T, NL, WK, CT, ROT, A4>                                     \
        <<<gk, 128, 0, stream>>>(                                              \
        (const T *)k, (int8_t *)k_int8, (float *)k_scale, anchor_ptr, Lk, C,   \
        H_kv, k_sc_per_h, k_stride_b, k_stride_h, k_stride_n);                 \
    cudaError_t k_error = cudaGetLastError();                                   \
    if (k_error != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("quant_k kernel launch failed: ") + \
                               cudaGetErrorString(k_error));                    \
  } while (0)

#define LAUNCH_FUSED(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    const int H_max = H_q > H_kv ? H_q : H_kv;                                 \
    dim3 g(q_oblk + k_oblk, H_max, B);                                         \
    quant_qk_fused<T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4>                     \
        <<<g, 128, 0, stream>>>(                                                \
        (const T *)q, (int8_t *)q_int8, (float *)q_scale, (const T *)k,        \
        (int8_t *)k_int8, (float *)k_scale, anchor_ptr, Lq, Lk, C,             \
        q_oblk, H_q, H_kv, q_sc_per_h, k_sc_per_h, q_stride_b, q_stride_h,     \
        q_stride_n, k_stride_b, k_stride_h, k_stride_n);                       \
    cudaError_t qk_error = cudaGetLastError();                                  \
    if (qk_error != cudaSuccess)                                                \
      throw std::runtime_error(                                                 \
          std::string("quant_qk fused kernel launch failed: ") +              \
          cudaGetErrorString(qk_error));                                        \
  } while (0)

#define LAUNCH_C64(T, ROT)                                                      \
  LAUNCH_FUSED(T, 4, 8, 128, 32, 64, 64, 1, ROT, false)

#define LAUNCH_C128_CTA64(T, ROT)                                              \
  LAUNCH_FUSED(T, 4, 8, 128, 32, 64, 64, 1, ROT, true)

#define LAUNCH_C128_CTA128(T, ROT)                                             \
  LAUNCH_FUSED(T, 4, 16, 128, 32, 128, 128, 1, ROT, true)

#define LAUNCH_C256_CTA64(T, ROT)                                              \
  LAUNCH_SPLIT(T, 2, 8, 128, 16, 64, 64, 2, ROT, true)

#define LAUNCH_C256_CTA128(T, ROT)                                             \
  LAUNCH_SPLIT(T, 2, 16, 128, 16, 128, 128, 2, ROT, true)

// Dispatch head dimension and CTA shape before rotation. The previous
// LAUNCH_SELECTED macro put runtime shape checks inside every rotation/dtype
// branch, which made NVCC instantiate all eight tile configurations at every
// call site even though only these five configurations can reach the kernel.
#define DISPATCH_C64(T, LAUNCHER)                                              \
  do {                                                                         \
    if (Lk <= 256) {                                                           \
      LAUNCHER(T, 4);                                                          \
    } else {                                                                   \
      LAUNCHER(T, 64);                                                         \
    }                                                                          \
  } while (0)

#define DISPATCH_C128(T, LAUNCHER)                                             \
  do {                                                                         \
    if (Lk <= 256) {                                                           \
      LAUNCHER(T, 4);                                                          \
    } else {                                                                   \
      LAUNCHER(T, 128);                                                        \
    }                                                                          \
  } while (0)

#define DISPATCH_C256(T, LAUNCHER)                                             \
  do {                                                                         \
    if (Lk <= 256) {                                                           \
      LAUNCHER(T, 4);                                                          \
    } else {                                                                   \
      LAUNCHER(T, 129);                                                        \
    }                                                                          \
  } while (0)

#define DO(T)                                                                  \
  do {                                                                         \
    if (C == 64) {                                                             \
      DISPATCH_C64(T, LAUNCH_C64);                                             \
    } else if (C == 128) {                                                     \
      if (BLKK == 128) {                                                       \
        DISPATCH_C128(T, LAUNCH_C128_CTA128);                                  \
      } else {                                                                 \
        DISPATCH_C128(T, LAUNCH_C128_CTA64);                                   \
      }                                                                        \
    } else if (BLKK == 128) {                                                  \
      DISPATCH_C256(T, LAUNCH_C256_CTA128);                                    \
    } else {                                                                   \
      DISPATCH_C256(T, LAUNCH_C256_CTA64);                                     \
    }                                                                          \
  } while (0)

  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] { DO(T); });

#undef LAUNCH_SPLIT
#undef LAUNCH_FUSED
#undef LAUNCH_C64
#undef LAUNCH_C128_CTA64
#undef LAUNCH_C128_CTA128
#undef LAUNCH_C256_CTA64
#undef LAUNCH_C256_CTA128
#undef DISPATCH_C64
#undef DISPATCH_C128
#undef DISPATCH_C256
#undef DO
}

void launch_quant_q_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, int B, int H_q, int Lq, int C,
    int full_Lk,
    int64_t q_stride_b, int64_t q_stride_h, int64_t q_stride_n,
    int input_dtype_code, cudaStream_t stream) {
  if (C != 128) {
    throw std::runtime_error(
        "quant_q_per_thread_int8: streamed H3 requires head_dim 128");
  }
  if (!q || !q_int8 || !q_scale || B <= 0 || H_q <= 0 || Lq <= 0) {
    throw std::runtime_error(
        "quant_q_per_thread_int8: invalid buffer or geometry");
  }
  if (full_Lk <= 0) {
    throw std::runtime_error(
        "quant_q_per_thread_int8: full K length must be positive");
  }

  const size_t element_size =
      input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const size_t vector_size = 4 * element_size;
  const auto stride_ok = [](int64_t stride, int extent) {
    return extent < 2 || (stride > 0 && stride % 4 == 0);
  };
  if (reinterpret_cast<uintptr_t>(q) % vector_size != 0 ||
      !stride_ok(q_stride_b, B) ||
      !stride_ok(q_stride_h, H_q) ||
      !stride_ok(q_stride_n, Lq)) {
    throw std::runtime_error(
        "quant_q_per_thread_int8: Q pointer/strides must preserve "
        "4-element alignment");
  }

  constexpr int BLKQ = 128;
  constexpr int WARPQ = 32;
  constexpr int CHANNEL_TILES = 1;
  const int q_oblk = (Lq + BLKQ - 1) / BLKQ * (BLKQ / WARPQ);
  const int q_sc_per_h = q_oblk * 8;
  dim3 grid(q_oblk, H_q, B);

  // Match the rotation used by the coupled C=128 Q/K quantizer.
  // Short full-K sequences use rotation 4; otherwise H3 uses rotation 128.
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    if (full_Lk <= 256) {
      quant_q_kernel<T, 4, BLKQ, WARPQ, CHANNEL_TILES, 4, true>
          <<<grid, 128, 0, stream>>>(
              static_cast<const T *>(q),
              static_cast<int8_t *>(q_int8),
              static_cast<float *>(q_scale),
              Lq, C, H_q, q_sc_per_h,
              q_stride_b, q_stride_h, q_stride_n);
    } else {
      quant_q_kernel<T, 4, BLKQ, WARPQ, CHANNEL_TILES, 128, true>
          <<<grid, 128, 0, stream>>>(
              static_cast<const T *>(q),
              static_cast<int8_t *>(q_int8),
              static_cast<float *>(q_scale),
              Lq, C, H_q, q_sc_per_h,
              q_stride_b, q_stride_h, q_stride_n);
    }
  });
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("quant_q kernel launch failed: ") +
        cudaGetErrorString(error));
  }
}

void launch_select_k_anchor_from_samples(
    const void *samples, const int *sample_positions, void *anchor_values,
    void *anchor_indices, int B, int H_kv, int full_Lk, int C,
    int64_t stride_b, int64_t stride_h, int64_t stride_n,
    int input_dtype_code, cudaStream_t stream) {
  if (B <= 0 || H_kv <= 0 || full_Lk <= 0 ||
      (C != 64 && C != 128 && C != 256)) {
    throw std::runtime_error(
        "select_k_anchor: invalid batch, head, sequence, or head_dim");
  }
  if (!samples || !sample_positions || !anchor_values || !anchor_indices) {
    throw std::runtime_error("select_k_anchor: all buffers are required");
  }
  dim3 grid(H_kv, B);
  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] {
    detect_k_anchor<T><<<grid, CENTER_DETECT_THREADS, 0, stream>>>(
        static_cast<const T *>(samples), static_cast<int *>(anchor_indices),
        full_Lk, C, H_kv, stride_b, stride_h, stride_n, sample_positions,
        static_cast<T *>(anchor_values));
  });
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("select_k_anchor kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}

void launch_quant_qk_per_thread_int8_into(
    const void *q, const void *k, void *q_int8, void *q_scale,
    void *k_int8, void *k_scale, const void *anchor_values,
    const void *anchor_indices, int B, int H_q, int Lq, int full_Lq,
    int q_start, int H_kv, int Lk, int full_Lk, int k_start, int C,
    int cta_k, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, cudaStream_t stream) {
  if (C != 64 && C != 128 && C != 256) {
    throw std::runtime_error(
        "quant_qk_into: padded head_dim must be 64, 128, or 256");
  }
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("quant_qk_into: cta_k must be 64 or 128");
  }
  if (C == 64 && cta_k != 64) {
    throw std::runtime_error("quant_qk_into: cta_k 128 is unsupported for head_dim 64");
  }
  if (B <= 0 || H_q <= 0 || H_kv <= 0 || Lq <= 0 || Lk <= 0 ||
      full_Lq <= 0 || full_Lk <= 0 || q_start < 0 || k_start < 0 ||
      q_start + Lq > full_Lq || k_start + Lk > full_Lk) {
    throw std::runtime_error("quant_qk_into: invalid shape or destination range");
  }
  if (q_start % 128 != 0 || k_start % cta_k != 0 ||
      (q_start + Lq < full_Lq && Lq % 128 != 0) ||
      (k_start + Lk < full_Lk && Lk % cta_k != 0)) {
    throw std::runtime_error("quant_qk_into: chunk ranges are not tile aligned");
  }
  if (!q || !k || !q_int8 || !q_scale || !k_int8 || !k_scale ||
      !anchor_values || !anchor_indices) {
    throw std::runtime_error("quant_qk_into: all buffers are required");
  }

  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const size_t vector_size = 4 * element_size;
  const auto stride_ok = [](int64_t stride, int extent) {
    return extent < 2 || (stride > 0 && stride % 4 == 0);
  };
  const auto is_vector_aligned =
      [vector_size, &stride_ok](const void *ptr, int64_t stride_b,
                                int64_t stride_h, int64_t stride_n,
                                int extent_b, int extent_h, int extent_n) {
        return reinterpret_cast<uintptr_t>(ptr) % vector_size == 0 &&
               stride_ok(stride_b, extent_b) &&
               stride_ok(stride_h, extent_h) &&
               stride_ok(stride_n, extent_n);
      };
  if (!is_vector_aligned(q, q_stride_b, q_stride_h, q_stride_n, B, H_q, Lq) ||
      !is_vector_aligned(k, k_stride_b, k_stride_h, k_stride_n, B, H_kv, Lk) ||
      reinterpret_cast<uintptr_t>(anchor_values) % vector_size != 0) {
    throw std::runtime_error(
        "quant_qk_into: Q/K/anchor pointers and strides must preserve 4-element alignment");
  }

  const int q_sc_per_h = ((full_Lq + 127) / 128) * (C == 256 ? 64 : 32);
  const int k_sc_per_h = ((full_Lk + cta_k - 1) / cta_k) * 4;

#define LAUNCH_INTO(T, NR, NL, BQ, WQ, BK, WK, CT, ROT, A4)                   \
  do {                                                                         \
    const int q_oblk = (Lq + BQ - 1) / BQ * (BQ / WQ);                        \
    const int k_oblk = (Lk + BK - 1) / BK * (BK / WK);                        \
    dim3 q_grid(q_oblk, H_q, B);                                               \
    quant_q_into_kernel<T, NR, BQ, WQ, CT, ROT, A4>                           \
        <<<q_grid, 128, 0, stream>>>(                                          \
            static_cast<const T *>(q), static_cast<int8_t *>(q_int8),         \
            static_cast<float *>(q_scale), Lq, full_Lq, q_start, C, H_q,      \
            q_sc_per_h, q_stride_b, q_stride_h, q_stride_n);                  \
    cudaError_t q_error = cudaGetLastError();                                  \
    if (q_error != cudaSuccess)                                                \
      throw std::runtime_error(std::string("quant_q_into kernel launch failed: ") + \
                               cudaGetErrorString(q_error));                   \
    dim3 k_grid(k_oblk, H_kv, B);                                              \
    quant_k_into_kernel<T, NL, BK, WK, CT, ROT, A4>                           \
        <<<k_grid, 128, 0, stream>>>(                                          \
            static_cast<const T *>(k), static_cast<int8_t *>(k_int8),         \
            static_cast<float *>(k_scale),                                    \
            static_cast<const T *>(anchor_values),                            \
            static_cast<const int *>(anchor_indices), Lk, full_Lk, k_start,   \
            C, H_kv, k_sc_per_h, k_stride_b, k_stride_h, k_stride_n);         \
    cudaError_t k_error = cudaGetLastError();                                  \
    if (k_error != cudaSuccess)                                                \
      throw std::runtime_error(std::string("quant_k_into kernel launch failed: ") + \
                               cudaGetErrorString(k_error));                   \
  } while (0)

#define DISPATCH_INTO(T)                                                       \
  do {                                                                         \
    if (C == 64) {                                                             \
      if (full_Lk <= 256)                                                      \
        LAUNCH_INTO(T, 4, 8, 128, 32, 64, 64, 1, 4, false);                   \
      else                                                                     \
        LAUNCH_INTO(T, 4, 8, 128, 32, 64, 64, 1, 64, false);                  \
    } else if (C == 128 && cta_k == 64) {                                     \
      if (full_Lk <= 256)                                                      \
        LAUNCH_INTO(T, 4, 8, 128, 32, 64, 64, 1, 4, true);                    \
      else                                                                     \
        LAUNCH_INTO(T, 4, 8, 128, 32, 64, 64, 1, 128, true);                  \
    } else if (C == 128) {                                                     \
      if (full_Lk <= 256)                                                      \
        LAUNCH_INTO(T, 4, 16, 128, 32, 128, 128, 1, 4, true);                 \
      else                                                                     \
        LAUNCH_INTO(T, 4, 16, 128, 32, 128, 128, 1, 128, true);               \
    } else if (cta_k == 64) {                                                  \
      if (full_Lk <= 256)                                                      \
        LAUNCH_INTO(T, 2, 8, 128, 16, 64, 64, 2, 4, true);                    \
      else                                                                     \
        LAUNCH_INTO(T, 2, 8, 128, 16, 64, 64, 2, 129, true);                  \
    } else {                                                                   \
      if (full_Lk <= 256)                                                      \
        LAUNCH_INTO(T, 2, 16, 128, 16, 128, 128, 2, 4, true);                 \
      else                                                                     \
        LAUNCH_INTO(T, 2, 16, 128, 16, 128, 128, 2, 129, true);               \
    }                                                                          \
  } while (0)

  DISPATCH_FP_DTYPE(input_dtype_code, T, [&] { DISPATCH_INTO(T); });

#undef LAUNCH_INTO
#undef DISPATCH_INTO
}
