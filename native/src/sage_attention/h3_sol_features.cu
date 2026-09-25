// SPDX-License-Identifier: Apache-2.0
//
// H3-routed Sol-style complement for the vendored Kitchen INT8 carrier.
//
// Routing remains entirely H3-owned.  This file consumes that route and adds:
//   * pooled omitted-block mass scored with Kitchen-compatible quantized
//     Q/K centroids,
//   * Sol-style two-pass histogram token augmentation over blocks omitted by
//     both neighbouring 64-row query tiles, and
//   * an in-place shared-softmax merge with the exact sparse Kitchen result.
//
// The token stage reads the existing Kitchen Q/K/V carriers directly; it does
// not keep a second BF16 or INT8 copy of Q/K.  The histogram policy matches
// Comfy Kitchen's public Sol implementation: 128 bins, 0.25-log2 bins from
// ref-8 through ref+16, then 2.0-log2 bins through ref+80, with whole-bin
// admission at the budget boundary.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int HD = 128;
constexpr int Q_TILE = 64;
constexpr int TOK_GROUP = 2;
constexpr int TOK_BUDGET_MAX = 256;
constexpr int HIST_BINS = 128;
constexpr int HIST_FINE_BINS = 96;
constexpr float HIST_LOW = 8.0f;
constexpr float HIST_FINE_WIDTH = 0.25f;
constexpr float HIST_COARSE_WIDTH = 2.0f;
constexpr int TOK_CHUNK = 128;
constexpr float NEG_INF = -3.0e38f;
constexpr float LOG2E_F = 1.4426950408889634f;

inline size_t align16(size_t n) { return (n + 15u) & ~(size_t)15u; }

__device__ __forceinline__ int inv_perm16(int w) {
  return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) |
         (((w >> 2) & 1) << 3);
}

__device__ __forceinline__ float atomic_max_float(float *addr, float value) {
  int *ptr = reinterpret_cast<int *>(addr);
  int old = *ptr;
  while (__int_as_float(old) < value) {
    const int assumed = old;
    old = atomicCAS(ptr, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

template <typename T> __device__ __forceinline__ float as_float(T x) {
  return static_cast<float>(x);
}
template <> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <> __device__ __forceinline__ float as_float<half>(half x) {
  return __half2float(x);
}

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}
template <> __device__ __forceinline__ half from_float<half>(float x) {
  return __float2half(x);
}

__device__ __forceinline__ float block_sum_128(float x) {
  __shared__ float warp_sum[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off; off >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, off);
  if (lane == 0) warp_sum[warp] = x;
  __syncthreads();
  float out = threadIdx.x < 4 ? warp_sum[lane] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int off = 16; off; off >>= 1)
      out += __shfl_down_sync(0xffffffffu, out, off);
  }
  __syncthreads();
  if (threadIdx.x == 0) warp_sum[0] = out;
  __syncthreads();
  return warp_sum[0];
}

__device__ __forceinline__ float block_max_128(float x) {
  __shared__ float warp_max[4];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off; off >>= 1)
    x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, off));
  if (lane == 0) warp_max[warp] = x;
  __syncthreads();
  float out = threadIdx.x < 4 ? warp_max[lane] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int off = 16; off; off >>= 1)
      out = fmaxf(out, __shfl_down_sync(0xffffffffu, out, off));
  }
  __syncthreads();
  if (threadIdx.x == 0) warp_max[0] = out;
  __syncthreads();
  return warp_max[0];
}

__device__ __forceinline__ int8_t quant_i8(float x, float inv_scale) {
  return (int8_t)max(-127, min(127, __float2int_rn(x * inv_scale)));
}

__device__ __forceinline__ bool route_bit(
    const int32_t *bits, int bhq, int qb, int kb, int nq, int words) {
  const uint32_t word = reinterpret_cast<const uint32_t *>(bits)[
      ((int64_t)bhq * nq + qb) * words + (kb >> 5)];
  return (word >> (kb & 31)) & 1u;
}

__device__ __forceinline__ float q_row_scale(
    const float *q_scale, int b, int h, int row, int H, int scales_per_head) {
  const int tile = row >> 7;           // 128-row Kitchen Q packing tile
  const int local = row & 127;
  const int sub = local >> 5;          // 32-row quantization subgroup
  const int residue = local & 7;       // shared by rows spaced by 8
  return q_scale[((int64_t)b * H + h) * scales_per_head + tile * 32 + sub * 8 + residue];
}

__device__ __forceinline__ float k_row_scale(
    const float *k_scale, int b, int h, int row, int H, int cta_k,
    int scales_per_head) {
  const int tile = row / cta_k;
  const int residue_group = (row & 7) >> 1;
  return k_scale[((int64_t)b * H + h) * scales_per_head + tile * 4 + residue_group];
}

__device__ __forceinline__ int v_storage_index(int token) {
  return (token & ~15) | inv_perm16(token & 15);
}

__device__ __forceinline__ float v_value(
    const int8_t *v, const float *v_scale, int b, int h, int token, int d,
    int H, int padded_k) {
  const int dst = v_storage_index(token);
  const int64_t row = ((int64_t)b * H + h) * HD + d;
  return (float)v[row * padded_k + dst] * v_scale[row];
}

__device__ __forceinline__ float hist_edge(int index) {
  if (index <= HIST_FINE_BINS)
    return (float)index * HIST_FINE_WIDTH;
  return HIST_FINE_BINS * HIST_FINE_WIDTH +
         (float)(index - HIST_FINE_BINS) * HIST_COARSE_WIDTH;
}

__device__ __forceinline__ int hist_bin(float relative) {
  const float fine_span = HIST_FINE_BINS * HIST_FINE_WIDTH;
  int bin;
  if (relative < fine_span) {
    bin = (int)floorf(relative / HIST_FINE_WIDTH);
  } else {
    bin = HIST_FINE_BINS + (int)floorf((relative - fine_span) / HIST_COARSE_WIDTH);
  }
  if (bin < 0) bin = 0;
  if (bin >= HIST_BINS) bin = HIST_BINS - 1;
  return bin;
}

__device__ __forceinline__ float sign128(int channel) {
  constexpr uint32_t s0 = 0x1035997bu;
  constexpr uint32_t s1 = 0x8087f5eeu;
  constexpr uint32_t s2 = 0xee2e4e1au;
  constexpr uint32_t s3 = 0x71132418u;
  const int lane = channel >> 2;
  const int c = channel & 3;
  const uint32_t signs = lane < 8 ? s0 : lane < 16 ? s1 : lane < 24 ? s2 : s3;
  const int shift = (lane & 7) * 4 + c;
  return ((signs >> shift) & 1u) ? 1.0f : -1.0f;
}

// Match Kitchen's HD128 Q/K transform dispatch: short sequences use the
// normalized H4 transform; long sequences use the randomized normalized H128.
// The transform is orthonormal, but quantizing after it makes using the same
// basis as the exact Kitchen carrier important for softmax-state calibration.
__device__ __forceinline__ float rotate_centroid_128(
    float value, int d, int rotate_full, float *scratch) {
  if (rotate_full) value *= sign128(d);
  scratch[d] = value;
  __syncthreads();

  if (rotate_full) {
    for (int stride = 1; stride < HD; stride <<= 1) {
      const int mate = d ^ stride;
      const float a = scratch[d];
      const float m = scratch[mate];
      __syncthreads();
      scratch[d] = (d & stride) ? (m - a) : (a + m);
      __syncthreads();
    }
    value = scratch[d] * 0.08838834764831845f; // 1/sqrt(128)
  } else {
    const int g = d & ~3;
    const float a = scratch[g + 0], b = scratch[g + 1];
    const float c = scratch[g + 2], e = scratch[g + 3];
    __syncthreads();
    const int j = d & 3;
    if (j == 0) value = (a + b + c + e) * 0.5f;
    else if (j == 1) value = (a - b + c - e) * 0.5f;
    else if (j == 2) value = (a + b - c - e) * 0.5f;
    else value = (a - b - c + e) * 0.5f;
  }
  __syncthreads();
  return value;
}

__global__ void build_route_bits_kernel(
    const int32_t *__restrict__ route, const int32_t *__restrict__ counts,
    int32_t *__restrict__ bits, int rows, int slots, int words,
    int route_is_delta) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  const int count = max(0, min(counts[row], slots));
  int value = 0;
  uint32_t *dst = reinterpret_cast<uint32_t *>(bits + (int64_t)row * words);
  for (int i = 0; i < count; ++i) {
    const int raw = route[(int64_t)row * slots + i];
    value = route_is_delta ? value + raw : raw;
    if (value >= 0 && value < words * 32)
      dst[value >> 5] |= 1u << (value & 31);
  }
}

// Precompute each centred K block centroid once.  The temporary carrier lives
// in the existing group_q allocation until pooled_tail_kernel completes; the
// token-stage group_q kernel overwrites that buffer afterwards.  This removes
// an O(NQ*NK) repeated centroid transform/quantize while preserving the exact
// same centroid score arithmetic.
template <typename SummaryT>
__global__ void precompute_k_centroid_kernel(
    const SummaryT *__restrict__ k_summary,
    const SummaryT *__restrict__ k_offset,
    int8_t *__restrict__ k_centroid,
    float *__restrict__ k_centroid_scale,
    int B, int Hkv, int NK, int rotate_full) {
  const int d = threadIdx.x;
  const int kb = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (d >= HD || kb >= NK) return;

  __shared__ float rotate_scratch[HD];
  __shared__ float scale;
  const int64_t kbase = (((int64_t)b * Hkv + h) * NK + kb) * HD;
  const int64_t koff = ((int64_t)b * Hkv + h) * HD;
  float k_rot = rotate_centroid_128(
      as_float(k_summary[kbase + d]) - as_float(k_offset[koff + d]),
      d, rotate_full, rotate_scratch);
  const float k_max = block_max_128(fabsf(k_rot));
  if (d == 0) scale = k_max / 127.0f + 1.0e-7f;
  __syncthreads();

  k_centroid[kbase + d] = quant_i8(k_rot, 1.0f / scale);
  if (d == 0)
    k_centroid_scale[((int64_t)b * Hkv + h) * NK + kb] = scale;
}

template <typename SummaryT>
__global__ void group_q_kernel(
    const SummaryT *__restrict__ q_summary, float *__restrict__ group_q,
    int B, int H, int NQ, int groups, int rotate_full) {
  const int d = threadIdx.x;
  const int group = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (d >= HD || group >= groups) return;

  __shared__ float x[HD];
  const int q0 = group * TOK_GROUP;
  const int q1 = q0 + 1;
  const int64_t base0 = (((int64_t)b * H + h) * NQ + q0) * HD;
  float value = as_float(q_summary[base0 + d]);
  if (q1 < NQ) {
    const int64_t base1 = (((int64_t)b * H + h) * NQ + q1) * HD;
    value = 0.5f * (value + as_float(q_summary[base1 + d]));
  }
  value = rotate_centroid_128(value, d, rotate_full, x);

  // Match the pooled centroid score system used for group_ref: quantize the
  // grouped Q centroid as its own pseudo Kitchen row, then store its
  // dequantized value for the existing token kernels.  This keeps token score
  // comparisons against group_ref in the same centroid-quantized Q space
  // without changing the token-stage ABI or buffers.
  const float qmax = block_max_128(fabsf(value));
  __shared__ float qscale;
  if (d == 0) qscale = qmax / 127.0f + 1.0e-7f;
  __syncthreads();
  const int8_t q8 = quant_i8(value, 1.0f / qscale);
  group_q[(((int64_t)b * H + h) * groups + group) * HD + d] =
      (float)q8 * qscale;
}

template <typename SummaryT>
__global__ void pooled_tail_kernel(
    const SummaryT *__restrict__ q_summary,
    const int8_t *__restrict__ k_centroid,
    const float *__restrict__ k_centroid_scale,
    const float *__restrict__ v_sum,
    const int32_t *__restrict__ bits,
    float *__restrict__ pooled_out, float *__restrict__ pooled_lse,
    float *__restrict__ group_ref,
    int B, int Hq, int Hkv, int NQ, int NK, int words, int Lk,
    int kv_tile, int token_budget, int rotate_full, float scale_log2) {
  const int d = threadIdx.x;
  const int qb = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (d >= HD || qb >= NQ) return;
  const int kv_groups = Hq / Hkv;
  const int kh = h / kv_groups;
  const int bhq = b * Hq + h;
  const int partner = (qb ^ 1) < NQ ? (qb ^ 1) : qb;
  const int group = qb >> 1;

  float numerator = 0.0f;
  __shared__ float sm_m, sm_l, sm_alpha, sm_p, sm_score;
  __shared__ float rotate_scratch[HD];
  __shared__ int8_t q_centroid[HD];
  __shared__ float q_centroid_scale;
  if (threadIdx.x == 0) { sm_m = NEG_INF; sm_l = 0.0f; }
  __syncthreads();
  float local_ref = NEG_INF;

  const int64_t qbase = (((int64_t)b * Hq + h) * NQ + qb) * HD;

  // Quantize the Q centroid as a pseudo Kitchen row.  The exact sparse branch
  // computes logits from Kitchen's rotated INT8 carrier, so keeping the pooled
  // centroid in the same transformed/quantized score system avoids merging a
  // full-precision centroid LSE against an INT8 exact LSE.
  float q_rot = rotate_centroid_128(
      as_float(q_summary[qbase + d]), d, rotate_full, rotate_scratch);
  const float q_max = block_max_128(fabsf(q_rot));
  if (threadIdx.x == 0)
    q_centroid_scale = q_max / 127.0f + 1.0e-7f;
  __syncthreads();
  q_centroid[d] = quant_i8(q_rot, 1.0f / q_centroid_scale);
  __syncthreads();

  for (int kb = 0; kb < NK; ++kb) {
    if (route_bit(bits, bhq, qb, kb, NQ, words)) continue;
    const bool partner_omits = !route_bit(bits, bhq, partner, kb, NQ, words);
    const bool token_candidate = token_budget > 0 && partner_omits;

    const int64_t kbase = (((int64_t)b * Hkv + kh) * NK + kb) * HD;
    const float ks = k_centroid_scale[((int64_t)b * Hkv + kh) * NK + kb];
    const float int_dot = block_sum_128(
        (float)q_centroid[d] * (float)k_centroid[kbase + d]);
    const float score = int_dot * q_centroid_scale * ks * scale_log2;
    if (threadIdx.x == 0) sm_score = score;
    __syncthreads();

    if (token_candidate) {
      if (threadIdx.x == 0) local_ref = fmaxf(local_ref, sm_score);
      __syncthreads();
      continue;
    }

    if (threadIdx.x == 0) {
      const float new_m = fmaxf(sm_m, sm_score);
      sm_alpha = isfinite(sm_m) ? exp2f(sm_m - new_m) : 0.0f;
      sm_p = exp2f(sm_score - new_m);
      const int len = min(kv_tile, Lk - kb * kv_tile);
      sm_l = sm_l * sm_alpha + sm_p * (float)max(len, 0);
      sm_m = new_m;
    }
    __syncthreads();
    const int64_t vbase = (((int64_t)b * Hkv + kh) * NK + kb) * HD;
    numerator = numerator * sm_alpha + sm_p * v_sum[vbase + d];
    __syncthreads();
  }

  const int64_t out_idx = (((int64_t)b * Hq + h) * NQ + qb) * HD + d;
  if (sm_l > 0.0f) pooled_out[out_idx] = numerator / sm_l;
  else pooled_out[out_idx] = 0.0f;
  if (threadIdx.x == 0) {
    pooled_lse[((int64_t)b * Hq + h) * NQ + qb] =
        sm_l > 0.0f ? sm_m + log2f(sm_l) : NEG_INF;
    if (isfinite(local_ref))
      atomic_max_float(&group_ref[((int64_t)b * Hq + h) * ((NQ + 1) >> 1) + group], local_ref);
  }
}

// Token-level histogram over H3-omitted candidate blocks.
__global__ void histogram_kernel_v2(
    const float *__restrict__ group_q,
    const int8_t *__restrict__ k, const float *__restrict__ k_scale,
    const int32_t *__restrict__ bits,
    const float *__restrict__ group_ref,
    int32_t *__restrict__ histogram, float *__restrict__ token_max,
    int B, int Hq, int Hkv, int Lk, int NQ, int groups, int words,
    int kv_tile, int cta_k, int k_scales_per_head, float scale_log2) {
  const int token = blockIdx.x * TOK_CHUNK + threadIdx.x;
  const int group = blockIdx.y;
  const int bhq = blockIdx.z;
  if (token >= Lk || group >= groups) return;
  const int b = bhq / Hq, h = bhq % Hq;
  const int kv_groups = Hq / Hkv, kh = h / kv_groups;
  const int qb0 = group * TOK_GROUP;
  const int qb1 = min(NQ - 1, qb0 + 1);
  const int kb = token / kv_tile;
  if (route_bit(bits, bhq, qb0, kb, NQ, words) ||
      route_bit(bits, bhq, qb1, kb, NQ, words)) return;
  const float ref = group_ref[(int64_t)bhq * groups + group];
  if (!isfinite(ref)) return;

  const float *qg = group_q + ((int64_t)bhq * groups + group) * HD;
  const float ks = k_row_scale(k_scale, b, kh, token, Hkv, cta_k, k_scales_per_head);
  const int8_t *krow = k + (((int64_t)b * Hkv + kh) * Lk + token) * HD;
  float dot = 0.0f;
  #pragma unroll
  for (int d = 0; d < HD; ++d) dot = fmaf(qg[d], (float)krow[d] * ks, dot);
  const float score = dot * scale_log2;
  atomic_max_float(&token_max[(int64_t)bhq * groups + group], score);
  const float rel = score - ref + HIST_LOW;
  if (rel >= 0.0f) {
    const int bin = hist_bin(rel);
    atomicAdd(&histogram[((int64_t)bhq * groups + group) * HIST_BINS + bin], 1);
  }
}

__global__ void threshold_kernel(
    const int32_t *__restrict__ histogram,
    const float *__restrict__ group_ref,
    float *__restrict__ threshold,
    int total_groups, int budget) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= total_groups) return;
  const float ref = group_ref[row];
  if (!isfinite(ref)) { threshold[row] = 3.0e38f; return; }
  int acc = 0;
  int boundary = -1;
  for (int b = HIST_BINS - 1; b >= 0; --b) {
    const int proposed = acc + histogram[(int64_t)row * HIST_BINS + b];
    if (proposed > budget) { boundary = b; break; }
    acc = proposed;
  }
  threshold[row] = ref - HIST_LOW + hist_edge(boundary + 1);
}

__global__ void token_tail_kernel(
    const float *__restrict__ group_q,
    const int8_t *__restrict__ k, const float *__restrict__ k_scale,
    const int8_t *__restrict__ v, const float *__restrict__ v_scale,
    const int32_t *__restrict__ bits,
    const float *__restrict__ threshold, const float *__restrict__ token_max,
    int32_t *__restrict__ selected_idx, int32_t *__restrict__ selected_count,
    float *__restrict__ token_num, float *__restrict__ token_den,
    int B, int Hq, int Hkv, int Lk, int padded_k, int NQ, int groups, int words,
    int kv_tile, int cta_k, int k_scales_per_head, int budget,
    float scale_log2) {
  const int split = blockIdx.x;
  const int group = blockIdx.y;
  const int bhq = blockIdx.z;
  const int token = split * TOK_CHUNK + threadIdx.x;
  const int b = bhq / Hq, h = bhq % Hq;
  const int kv_groups = Hq / Hkv, kh = h / kv_groups;
  const int qb0 = group * TOK_GROUP;
  const int qb1 = min(NQ - 1, qb0 + 1);
  const int64_t grow = (int64_t)bhq * groups + group;

  __shared__ float weights[TOK_CHUNK];
  float weight = 0.0f;
  if (token < Lk) {
    const int kb = token / kv_tile;
    const bool candidate =
        !route_bit(bits, bhq, qb0, kb, NQ, words) &&
        !route_bit(bits, bhq, qb1, kb, NQ, words);
    const float mx = token_max[grow];
    if (candidate && isfinite(mx)) {
      const float *qg = group_q + grow * HD;
      const float ks = k_row_scale(k_scale, b, kh, token, Hkv, cta_k, k_scales_per_head);
      const int8_t *krow = k + (((int64_t)b * Hkv + kh) * Lk + token) * HD;
      float dot = 0.0f;
      #pragma unroll
      for (int d = 0; d < HD; ++d) dot = fmaf(qg[d], (float)krow[d] * ks, dot);
      const float score = dot * scale_log2;
      if (score >= threshold[grow]) {
        const int slot = atomicAdd(&selected_count[grow], 1);
        if (slot < budget) {
          selected_idx[grow * budget + slot] = token;
        } else {
          // The histogram policy should keep whole-bin admission within the
          // budget.  Keep this overflow guard so an extreme score above the
          // final coarse bin never disappears from both branches.
          weight = exp2f(score - mx);
        }
      } else {
        weight = exp2f(score - mx);
      }
    }
  }
  weights[threadIdx.x] = weight;
  __syncthreads();

  const float local_den = block_sum_128(weight);
  if (threadIdx.x == 0 && local_den != 0.0f)
    atomicAdd(&token_den[grow], local_den);
  __syncthreads();

  const int d = threadIdx.x;
  float acc = 0.0f;
  if (d < HD) {
    #pragma unroll 4
    for (int j = 0; j < TOK_CHUNK; ++j) {
      const float w = weights[j];
      if (w == 0.0f) continue;
      const int t = split * TOK_CHUNK + j;
      if (t < Lk) acc = fmaf(w, v_value(v, v_scale, b, kh, t, d, Hkv, padded_k), acc);
    }
    if (acc != 0.0f) atomicAdd(&token_num[grow * HD + d], acc);
  }
}

__global__ void sort_selected_kernel(
    int32_t *__restrict__ selected_idx, int32_t *__restrict__ selected_count,
    int rows, int budget) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  int n = min(max(selected_count[row], 0), budget);
  selected_count[row] = n;
  int32_t *values = selected_idx + (int64_t)row * budget;
  for (int i = 1; i < n; ++i) {
    const int32_t x = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > x) { values[j + 1] = values[j]; --j; }
    values[j + 1] = x;
  }
}

template <typename OutT>
__global__ void final_merge_kernel(
    OutT *__restrict__ output, float *__restrict__ exact_lse,
    const int8_t *__restrict__ q, const int8_t *__restrict__ k,
    const int8_t *__restrict__ v,
    const float *__restrict__ q_scale, const float *__restrict__ k_scale,
    const float *__restrict__ v_scale,
    const float *__restrict__ pooled_out, const float *__restrict__ pooled_lse,
    const float *__restrict__ token_num, const float *__restrict__ token_den,
    const float *__restrict__ token_max,
    const int32_t *__restrict__ selected_idx, const int32_t *__restrict__ selected_count,
    int B, int Hq, int Hkv, int Lq, int Lk, int padded_k, int NQ, int groups,
    int cta_k, int q_scales_per_head, int k_scales_per_head, int budget,
    int64_t out_sb, int64_t out_sh, int64_t out_sn,
    float scale_log2) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  if (row >= Lq || warp >= 8) return;
  const int kv_groups = Hq / Hkv, kh = h / kv_groups;
  const int qb = row / Q_TILE;
  const int group = qb >> 1;
  const int64_t grow = ((int64_t)b * Hq + h) * groups + group;
  const int nsel = min(selected_count[grow], budget);
  const int c0 = lane * 4;
  const float qs = q_row_scale(q_scale, b, h, row, Hq, q_scales_per_head);
  const int8_t *qrow = q + (((int64_t)b * Hq + h) * Lq + row) * HD;

  float sel_m = NEG_INF, sel_l = 0.0f;
  float sel_num[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < nsel; ++i) {
    const int token = selected_idx[grow * budget + i];
    if (token < 0 || token >= Lk) continue;
    const float ks = k_row_scale(k_scale, b, kh, token, Hkv, cta_k, k_scales_per_head);
    const int8_t *krow = k + (((int64_t)b * Hkv + kh) * Lk + token) * HD;
    float part = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      part += (float)qrow[c0 + j] * (float)krow[c0 + j];
    #pragma unroll
    for (int off = 16; off; off >>= 1)
      part += __shfl_down_sync(0xffffffffu, part, off);
    const float dot = __shfl_sync(0xffffffffu, part, 0);
    const float score = dot * qs * ks * scale_log2;
    const float new_m = fmaxf(sel_m, score);
    const float alpha = isfinite(sel_m) ? exp2f(sel_m - new_m) : 0.0f;
    const float p = exp2f(score - new_m);
    sel_l = sel_l * alpha + p;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      sel_num[j] = sel_num[j] * alpha + p * v_value(v, v_scale, b, kh, token, c0 + j, Hkv, padded_k);
    sel_m = new_m;
  }
  const float sel_lse = sel_l > 0.0f ? sel_m + log2f(sel_l) : NEG_INF;

  const int64_t lse_idx = ((int64_t)b * Hq + h) * Lq + row;
  const float e_lse = exact_lse[lse_idx];
  const float p_lse = pooled_lse[((int64_t)b * Hq + h) * NQ + qb];
  const float t_den = token_den[grow];
  const float t_lse = t_den > 0.0f && isfinite(token_max[grow])
                          ? token_max[grow] + log2f(t_den)
                          : NEG_INF;
  const float total_m = fmaxf(fmaxf(e_lse, p_lse), fmaxf(t_lse, sel_lse));
  const float we = isfinite(e_lse) ? exp2f(e_lse - total_m) : 0.0f;
  const float wp = isfinite(p_lse) ? exp2f(p_lse - total_m) : 0.0f;
  const float wt = isfinite(t_lse) ? exp2f(t_lse - total_m) : 0.0f;
  const float ws = isfinite(sel_lse) ? exp2f(sel_lse - total_m) : 0.0f;
  const float denom = we + wp + wt + ws;

  const int64_t obase = (int64_t)b * out_sb + (int64_t)h * out_sh + (int64_t)row * out_sn;
  const int64_t pbase = (((int64_t)b * Hq + h) * NQ + qb) * HD;
  const int64_t tbase = grow * HD;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int d = c0 + j;
    const float eo = as_float(output[obase + d]);
    const float po = pooled_out[pbase + d];
    const float to = t_den > 0.0f ? token_num[tbase + d] / t_den : 0.0f;
    const float so = sel_l > 0.0f ? sel_num[j] / sel_l : 0.0f;
    const float merged = denom > 0.0f ? (we * eo + wp * po + wt * to + ws * so) / denom : eo;
    output[obase + d] = from_float<OutT>(merged);
  }
  if (lane == 0 && denom > 0.0f)
    exact_lse[lse_idx] = total_m + log2f(denom);
}

} // namespace

void launch_h3_sol_features(
    void *output, void *exact_lse,
    const void *q, const void *k, const void *v,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *q_summary, const void *k_summary, const void *v_sum,
    const void *k_offset,
    const void *route, const void *counts,
    void *route_bits, void *group_q, void *pooled_out, void *pooled_lse,
    void *group_ref, void *histogram, void *token_max, void *threshold,
    void *selected_idx, void *selected_count, void *token_num, void *token_den,
    int B, int Hq, int Hkv, int Lq, int Lk, int padded_k,
    int NQ, int NK, int groups, int words, int route_slots,
    int route_is_delta, int kv_tile, int cta_k, int token_budget,
    int q_scales_per_head, int k_scales_per_head,
    int summary_dtype_code, int output_dtype_code,
    int64_t out_sb, int64_t out_sh, int64_t out_sn,
    float attention_scale, cudaStream_t stream) {
  if (Hq <= 0 || Hkv <= 0 || Hq % Hkv != 0)
    throw std::runtime_error("h3_sol_features: invalid Q/KV head geometry");
  if (NQ != (Lq + Q_TILE - 1) / Q_TILE)
    throw std::runtime_error("h3_sol_features: Sol token augmentation requires 64-row H3 query tiles");
  if (kv_tile != 64 && kv_tile != 128)
    throw std::runtime_error("h3_sol_features: KV tile must be 64 or 128");
  if (NK != (Lk + kv_tile - 1) / kv_tile)
    throw std::runtime_error("h3_sol_features: K summary geometry does not match KV tile");
  if (cta_k != 64 && cta_k != 128)
    throw std::runtime_error("h3_sol_features: Kitchen carrier CTA_K must be 64 or 128");
  if (token_budget < 0 || token_budget > TOK_BUDGET_MAX)
    throw std::runtime_error("h3_sol_features: token budget must be in [0,256]");
  if (!output || !exact_lse || !q || !k || !v || !q_scale || !k_scale || !v_scale ||
      !q_summary || !k_summary || !v_sum || !k_offset || !route || !counts ||
      !route_bits || !group_q || !pooled_out || !pooled_lse || !group_ref)
    throw std::runtime_error("h3_sol_features: null required pointer");

  const int route_rows = B * Hq * NQ;
  build_route_bits_kernel<<<route_rows, 1, 0, stream>>>(
      static_cast<const int32_t *>(route), static_cast<const int32_t *>(counts),
      static_cast<int32_t *>(route_bits), route_rows, route_slots, words,
      route_is_delta);

  const int rotate_full = Lk > 256 ? 1 : 0;

  // Reuse group_q as a temporary K-centroid carrier.  pooled_tail_kernel is
  // ordered after this precompute on the same stream, and the token-stage
  // group_q kernel is ordered after pooled_tail_kernel, so the buffer can be
  // safely overwritten without any extra allocation or ABI change.
  const size_t group_q_bytes =
      (size_t)B * Hq * groups * HD * sizeof(float);
  const size_t k_centroid_count = (size_t)B * Hkv * NK;
  const size_t k_centroid_bytes = k_centroid_count * HD * sizeof(int8_t);
  const size_t k_scale_offset = align16(k_centroid_bytes);
  const size_t k_scratch_bytes =
      k_scale_offset + k_centroid_count * sizeof(float);
  if (k_scratch_bytes > group_q_bytes)
    throw std::runtime_error(
        "h3_sol_features: group_q scratch is too small for precomputed K centroids");

  char *centroid_scratch = static_cast<char *>(group_q);
  int8_t *k_centroid = reinterpret_cast<int8_t *>(centroid_scratch);
  float *k_centroid_scale =
      reinterpret_cast<float *>(centroid_scratch + k_scale_offset);

#define LAUNCH_SUMMARY(T) \
  precompute_k_centroid_kernel<T><<<dim3(NK, Hkv, B), HD, 0, stream>>>( \
      static_cast<const T *>(k_summary), static_cast<const T *>(k_offset), \
      k_centroid, k_centroid_scale, B, Hkv, NK, rotate_full); \
  pooled_tail_kernel<T><<<dim3(NQ, Hq, B), HD, 0, stream>>>( \
      static_cast<const T *>(q_summary), k_centroid, k_centroid_scale, \
      static_cast<const float *>(v_sum), static_cast<const int32_t *>(route_bits), \
      static_cast<float *>(pooled_out), static_cast<float *>(pooled_lse), \
      static_cast<float *>(group_ref), B, Hq, Hkv, NQ, NK, words, Lk, kv_tile, \
      token_budget, rotate_full, attention_scale * LOG2E_F); \
  if (token_budget > 0) \
    group_q_kernel<T><<<dim3(groups, Hq, B), HD, 0, stream>>>( \
        static_cast<const T *>(q_summary), static_cast<float *>(group_q), \
        B, Hq, NQ, groups, rotate_full)
  if (summary_dtype_code == 0) { LAUNCH_SUMMARY(float); }
  else if (summary_dtype_code == 1) { LAUNCH_SUMMARY(half); }
  else if (summary_dtype_code == 2) { LAUNCH_SUMMARY(__nv_bfloat16); }
  else throw std::runtime_error("h3_sol_features: unsupported summary dtype");
#undef LAUNCH_SUMMARY

  if (token_budget > 0) {
    if (!histogram || !token_max || !threshold || !selected_idx || !selected_count ||
        !token_num || !token_den)
      throw std::runtime_error("h3_sol_features: token augmentation scratch is incomplete");
    const int splits = (Lk + TOK_CHUNK - 1) / TOK_CHUNK;
    dim3 token_grid(splits, groups, B * Hq);
    histogram_kernel_v2<<<token_grid, TOK_CHUNK, 0, stream>>>(
        static_cast<const float *>(group_q), static_cast<const int8_t *>(k),
        static_cast<const float *>(k_scale), static_cast<const int32_t *>(route_bits),
        static_cast<const float *>(group_ref), static_cast<int32_t *>(histogram),
        static_cast<float *>(token_max), B, Hq, Hkv, Lk, NQ, groups, words,
        kv_tile, cta_k, k_scales_per_head, attention_scale * LOG2E_F);
    const int total_groups = B * Hq * groups;
    threshold_kernel<<<(total_groups + 255) / 256, 256, 0, stream>>>(
        static_cast<const int32_t *>(histogram), static_cast<const float *>(group_ref),
        static_cast<float *>(threshold), total_groups, token_budget);
    token_tail_kernel<<<token_grid, TOK_CHUNK, 0, stream>>>(
        static_cast<const float *>(group_q), static_cast<const int8_t *>(k),
        static_cast<const float *>(k_scale), static_cast<const int8_t *>(v),
        static_cast<const float *>(v_scale), static_cast<const int32_t *>(route_bits),
        static_cast<const float *>(threshold), static_cast<const float *>(token_max),
        static_cast<int32_t *>(selected_idx), static_cast<int32_t *>(selected_count),
        static_cast<float *>(token_num), static_cast<float *>(token_den),
        B, Hq, Hkv, Lk, padded_k, NQ, groups, words, kv_tile, cta_k,
        k_scales_per_head, token_budget, attention_scale * LOG2E_F);
    sort_selected_kernel<<<total_groups, 1, 0, stream>>>(
        static_cast<int32_t *>(selected_idx), static_cast<int32_t *>(selected_count),
        total_groups, token_budget);
  }

  dim3 merge_grid((Lq + 7) / 8, Hq, B);
#define LAUNCH_MERGE(T) \
  final_merge_kernel<T><<<merge_grid, 256, 0, stream>>>( \
      static_cast<T *>(output), static_cast<float *>(exact_lse), \
      static_cast<const int8_t *>(q), static_cast<const int8_t *>(k), \
      static_cast<const int8_t *>(v), static_cast<const float *>(q_scale), \
      static_cast<const float *>(k_scale), static_cast<const float *>(v_scale), \
      static_cast<const float *>(pooled_out), static_cast<const float *>(pooled_lse), \
      static_cast<const float *>(token_num), static_cast<const float *>(token_den), \
      static_cast<const float *>(token_max), static_cast<const int32_t *>(selected_idx), \
      static_cast<const int32_t *>(selected_count), B, Hq, Hkv, Lq, Lk, padded_k, \
      NQ, groups, cta_k, q_scales_per_head, k_scales_per_head, token_budget, \
      out_sb, out_sh, out_sn, attention_scale * LOG2E_F)
  if (output_dtype_code == 1) { LAUNCH_MERGE(half); }
  else if (output_dtype_code == 2) { LAUNCH_MERGE(__nv_bfloat16); }
  else throw std::runtime_error("h3_sol_features: output must be fp16 or bf16");
#undef LAUNCH_MERGE

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("h3_sol_features CUDA launch failed: ") + cudaGetErrorString(error));
}