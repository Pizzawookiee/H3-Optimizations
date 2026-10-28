// SPDX-License-Identifier: Apache-2.0
//
// Threshold routing for the streamed H3 sparse Kitchen path.
//
// A route row (one 64-token Q tile of one head) keeps video KV tile j when
// q . k_j > q . kbar + tau * sqrt(q^T Sigma q): the row's score mean plus tau
// standard deviations, where kbar and Sigma are the mean and population
// covariance of the video K-tile summaries. Those are exact statistics of the
// row's score distribution, computed from the Q summary alone, so no score
// matrix is materialized and there is no top-k or sort. tau comes from the KV
// budget through the Gaussian quantile (see the Python side).
//
// Two kernels per route chunk:
//   1. select: one CTA = one head x 64 Q rows (4 warps x 16 rows) x one segment
//      of the video KV range, so a chunk launches several CTAs per head. Sigma
//      (128 x 128 BF16) is staged through the two K buffers and each warp forms
//      Sigma q and q . kbar on the tensor cores, giving its rows' thresholds.
//      The segment's K summaries then stream through a two-stage cp.async
//      pipeline, 64 tiles per stage; each warp scores its 16 rows (m16n8k16
//      BF16 MMA, FP32 accumulate), compares in registers, and the four lanes
//      holding a row OR their bits into that row's 64-bit mask for the stage,
//      stored to a scratch buffer.
//   2. write: one warp per route row walks its stage masks in order; popcount
//      gives each selected tile its rank and count-leading-zeros its
//      predecessor, so the delta route comes out in ascending order. Dense rows
//      (Q tiles before the pure-video region) get the full dense route; sparse
//      rows get the dense context prefix, then the selected video tiles.
// BF16 MMA needs SM80; the Python side keeps top-k routing elsewhere.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int kDim = 128;
constexpr int kChunks = kDim * 2 / 16;  // 16-byte chunks per summary row
constexpr int kWarps = 4;
constexpr int kRowsPerCta = kWarps * 16;
constexpr int kStageRows = 64;
constexpr int kThreads = kWarps * 32;
constexpr int kRowBytes = kChunks * 16;
constexpr int kQBytes = kRowsPerCta * kRowBytes;
constexpr int kStageBytes = kStageRows * kRowBytes;
constexpr size_t kSmemBytes = kQBytes + 2 * kStageBytes + kDim * sizeof(float);

__device__ __forceinline__ uint32_t smem_u32(const void *pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

// Byte offset of 16-byte chunk c of row r. The XOR with the row's low bits
// puts the eight rows of an ldmatrix phase in distinct bank groups.
__device__ __forceinline__ uint32_t swizzled(int row, int chunk) {
  return static_cast<uint32_t>((row * kChunks + (chunk ^ (row & 7))) * 16);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void *src, bool pred) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int bytes = pred ? 16 : 0;  // zero-fill past the end
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
#else
  __trap();  // threshold routing is gated to SM80+ on the Python side
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t address) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  __trap();  // gated to SM80+ on the Python side
#endif
}

// Copy `rows` summary rows (128 BF16 each, row stride in elements) into a
// swizzled buffer; rows at or past `limit` are zero-filled.
__device__ __forceinline__ void load_rows(uint32_t buffer, const __nv_bfloat16 *source,
                                          int64_t row_stride, int first, int limit, int rows) {
  for (int index = threadIdx.x; index < rows * kChunks; index += kThreads) {
    const int row = index / kChunks;
    const int chunk = index % kChunks;
    const bool live = first + row < limit;
    const __nv_bfloat16 *src = source + (live ? (int64_t)(first + row) * row_stride : 0) + chunk * 8;
    cp_async16(buffer + swizzled(row, chunk), src, live);
  }
}

// acc[j] = this warp's 16 rows against stage rows 8j..8j+7 (all 128 dims).
__device__ __forceinline__ void score_stage(float (&acc)[8][4], const uint32_t (&a)[8][4],
                                            uint32_t stage, int lane) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      uint32_t b[4];
      const int row = j * 8 + (lane & 7) + ((lane >> 4) << 3);
      const int chunk = 2 * kk + ((lane >> 3) & 1);
      ldmatrix_x4(b, stage + swizzled(row, chunk));
      mma_bf16(acc[j], a[kk], b[0], b[1]);
      mma_bf16(acc[j + 1], a[kk], b[2], b[3]);
    }
  }
}

__device__ __forceinline__ float q_element(const unsigned char *q_smem, int row, int column) {
  const __nv_bfloat16 *chunk = reinterpret_cast<const __nv_bfloat16 *>(
      q_smem + swizzled(row, column >> 3));
  return __bfloat162float(chunk[column & 7]);
}

__global__ void __launch_bounds__(kThreads) threshold_select_kernel(
    const __nv_bfloat16 *__restrict__ q, const __nv_bfloat16 *__restrict__ k_video,
    const __nv_bfloat16 *__restrict__ sigma, const float *__restrict__ kbar,
    uint64_t *__restrict__ masks, int tile_count, int64_t q_bh_stride, int64_t q_row_stride,
    int64_t k_bh_stride, int video_tiles, int stages, int stages_per_segment, int dense_rows,
    float tau) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __trap();
#else
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char *q_smem = smem;
  const uint32_t q_base = smem_u32(q_smem);
  const uint32_t stage_base[2] = {q_base + kQBytes, q_base + kQBytes + kStageBytes};
  float *kbar_smem = reinterpret_cast<float *>(smem + kQBytes + 2 * kStageBytes);

  const int segment = blockIdx.x;
  const int row0 = blockIdx.y * kRowsPerCta;
  const int bh = blockIdx.z;
  if (row0 + kRowsPerCta <= dense_rows) return;  // every row of the group is dense
  const int first_stage = segment * stages_per_segment;
  const int last_stage = min(stages, first_stage + stages_per_segment);
  if (first_stage >= last_stage) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16 *q_head = q + bh * q_bh_stride;
  const __nv_bfloat16 *k_head = k_video + bh * k_bh_stride;
  const __nv_bfloat16 *sigma_head = sigma + (int64_t)bh * kDim * kDim;

  // Q rows and Sigma (both halves, one per stage buffer), then kbar.
  load_rows(q_base, q_head, q_row_stride, row0, tile_count, kRowsPerCta);
  load_rows(stage_base[0], sigma_head, kDim, 0, kDim, kStageRows);
  load_rows(stage_base[1], sigma_head, kDim, kStageRows, kDim, kStageRows);
  cp_async_commit();
  if (threadIdx.x < kDim) kbar_smem[threadIdx.x] = kbar[(int64_t)bh * kDim + threadIdx.x];
  cp_async_wait<0>();
  __syncthreads();

  uint32_t a[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const int row = warp * 16 + (lane & 15);
    ldmatrix_x4(a[kk], q_base + swizzled(row, 2 * kk + (lane >> 4)));
  }

  // Row statistics: Sigma q on the tensor cores, then q^T (Sigma q) and q . kbar.
  const int g = lane >> 2;
  const int local[2] = {warp * 16 + g, warp * 16 + g + 8};
  float variance[2] = {0.f, 0.f};
  float mean[2] = {0.f, 0.f};
  float acc[8][4];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    score_stage(acc, a, stage_base[half], lane);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int which = e >> 1;
        const int column = half * 64 + j * 8 + 2 * (lane & 3) + (e & 1);
        const float qv = q_element(q_smem, local[which], column);
        variance[which] += acc[j][e] * qv;
        mean[which] += kbar_smem[column] * qv;
      }
    }
  }
  float threshold[2];
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    variance[which] += __shfl_xor_sync(0xffffffffu, variance[which], 1);
    variance[which] += __shfl_xor_sync(0xffffffffu, variance[which], 2);
    mean[which] += __shfl_xor_sync(0xffffffffu, mean[which], 1);
    mean[which] += __shfl_xor_sync(0xffffffffu, mean[which], 2);
    threshold[which] = mean[which] + tau * sqrtf(fmaxf(variance[which], 0.f));
  }
  __syncthreads();  // Sigma buffers are about to be reused for K

  uint64_t *mask_out[2];
  bool live[2];
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    const int row = row0 + local[which];
    live[which] = row < tile_count && row >= dense_rows;
    mask_out[which] = masks + ((int64_t)bh * tile_count + min(row, tile_count - 1)) * stages;
  }

  load_rows(stage_base[0], k_head, kDim, first_stage * kStageRows, video_tiles, kStageRows);
  cp_async_commit();
  for (int s = first_stage; s < last_stage; ++s) {
    const int slot = (s - first_stage) & 1;
    if (s + 1 < last_stage) {
      load_rows(stage_base[slot ^ 1], k_head, kDim, (s + 1) * kStageRows, video_tiles, kStageRows);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    score_stage(acc, a, stage_base[slot], lane);

    const int first = s * kStageRows;
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      uint64_t mask = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int column = j * 8 + 2 * (lane & 3) + e;
          if (first + column < video_tiles && acc[j][which * 2 + e] > threshold[which]) {
            mask |= 1ull << column;
          }
        }
      }
      mask |= __shfl_xor_sync(0xffffffffu, mask, 1);
      mask |= __shfl_xor_sync(0xffffffffu, mask, 2);
      if (live[which] && (lane & 3) == 0) mask_out[which][s] = mask;
    }
    __syncthreads();  // all consumers finished before this buffer is refilled
  }
#endif
}

// One warp per route row: prefix, then the selected tiles as a delta route.
__global__ void __launch_bounds__(kThreads) threshold_write_kernel(
    const uint64_t *__restrict__ masks, int32_t *__restrict__ lut, int32_t *__restrict__ valid,
    int rows_total, int tile_count, int stages, int pvs, int kv_tiles, int dense_rows,
    int width) {
  const int lane = threadIdx.x & 31;
  const int flat = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (flat >= rows_total) return;
  const int row = flat % tile_count;
  int32_t *out = lut + (int64_t)flat * width;
  if (row < dense_rows) {
    for (int index = lane; index < kv_tiles; index += 32) out[index] = index == 0 ? 0 : 1;
    if (lane == 0) valid[flat] = kv_tiles;
    return;
  }
  for (int index = lane; index < pvs; index += 32) out[index] = index == 0 ? 0 : 1;
  const uint64_t *row_masks = masks + (int64_t)flat * stages;
  int count = 0;
  int last = max(pvs - 1, 0);
  for (int s0 = 0; s0 < stages; s0 += 32) {
    const uint64_t mine = s0 + lane < stages ? row_masks[s0 + lane] : 0ull;
    const int batch = min(32, stages - s0);
    for (int i = 0; i < batch; ++i) {
      const uint64_t mask = __shfl_sync(0xffffffffu, mine, i);
      if (!mask) continue;
      const int base = pvs + (s0 + i) * kStageRows;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int column = lane + 32 * half;
        if ((mask >> column) & 1ull) {
          const uint64_t below = mask & ((1ull << column) - 1ull);
          const int previous = below ? base + 63 - __clzll(below) : last;
          out[pvs + count + __popcll(below)] = base + column - previous;
        }
      }
      count += __popcll(mask);
      last = base + 63 - __clzll(mask);
    }
  }
  if (lane == 0) valid[flat] = pvs + count;
}

}  // namespace

int threshold_route_stages(int video_tiles) { return (video_tiles + kStageRows - 1) / kStageRows; }

void launch_h3_threshold_route(const void *q, const void *k_video, const void *sigma,
                               const void *kbar, void *lut, void *valid, void *masks, int bh,
                               int tile_count, int64_t q_bh_stride, int64_t q_row_stride,
                               int64_t k_bh_stride, int video_tiles, int pvs, int kv_tiles,
                               int dense_rows, int width, float tau, cudaStream_t stream) {
  if (bh <= 0 || tile_count <= 0) return;
  if (video_tiles <= 0 || pvs < 0 || kv_tiles != pvs + video_tiles || width < kv_tiles ||
      dense_rows < 0 || q_row_stride % 8 != 0 || q_bh_stride % 8 != 0 || k_bh_stride % 8 != 0) {
    throw std::runtime_error("threshold route received invalid geometry");
  }
  const int stages = threshold_route_stages(video_tiles);
  if (dense_rows < tile_count) {
    // Enough CTAs to cover every SM a few times over: split the KV stages.
    int device = 0;
    int sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int groups = (tile_count + kRowsPerCta - 1) / kRowsPerCta;
    const int wanted = max(1, (4 * max(sms, 1) + bh * groups - 1) / (bh * groups));
    const int per_segment = max(4, (stages + wanted - 1) / wanted);
    const int segments = (stages + per_segment - 1) / per_segment;
    cudaError_t error = cudaFuncSetAttribute(threshold_select_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(kSmemBytes));
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("threshold route smem request failed: ") +
                               cudaGetErrorString(error));
    }
    dim3 grid(segments, groups, bh);
    threshold_select_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k_video),
        static_cast<const __nv_bfloat16 *>(sigma), static_cast<const float *>(kbar),
        static_cast<uint64_t *>(masks), tile_count, q_bh_stride, q_row_stride, k_bh_stride,
        video_tiles, stages, per_segment, dense_rows, tau);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("threshold route select launch failed: ") +
                               cudaGetErrorString(error));
    }
  }
  const int rows_total = bh * tile_count;
  threshold_write_kernel<<<(rows_total + kWarps - 1) / kWarps, kThreads, 0, stream>>>(
      static_cast<const uint64_t *>(masks), static_cast<int32_t *>(lut),
      static_cast<int32_t *>(valid), rows_total, tile_count, stages, pvs, kv_tiles, dense_rows,
      width);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("threshold route write launch failed: ") +
                             cudaGetErrorString(error));
  }
}
