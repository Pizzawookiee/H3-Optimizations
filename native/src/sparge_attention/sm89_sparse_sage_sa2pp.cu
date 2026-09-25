// SPDX-License-Identifier: Apache-2.0
//
// Stream-aware H3 launcher for SpargeAttention's SM89 sparse FP8-V kernel.

#include "qattn/qk_int_sv_f8_cuda_sm89.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

constexpr uint32_t CTA_Q = 128;
constexpr uint32_t CTA_K = 64;
constexpr uint32_t WARP_Q = 32;
constexpr uint32_t WARP_K = 64;
constexpr uint32_t HEAD_DIM = 128;

template <typename DTypeOut>
void launch_impl(
    const int8_t *q, const int8_t *k, const int8_t *v, DTypeOut *output,
    const int32_t *lut, const int32_t *valid, const float *pv_threshold,
    const float *q_scale, const float *k_scale, const float *v_scale,
    uint32_t batch, uint32_t q_length, uint32_t kv_length,
    uint32_t q_heads, uint32_t kv_heads,
    uint32_t q_stride_b, uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t k_stride_b, uint32_t k_stride_n, uint32_t k_stride_h,
    uint32_t v_stride_b, uint32_t v_stride_h, uint32_t v_stride_d,
    uint32_t o_stride_b, uint32_t o_stride_n, uint32_t o_stride_h,
    float scale, cudaStream_t stream) {
  auto kernel = qk_int_sv_f8_block_sparse_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
      QuantGranularity::kPerBlock, QuantGranularity::kPerBlock, float, true,
      true, PVThresholdMode::kPerBlock, DTypeOut, ComputeUnit::kCudaCore,
      MaskMode::kNone, true, false>;
  constexpr size_t smem = CTA_Q * HEAD_DIM * sizeof(int8_t)
      + CTA_K * HEAD_DIM * sizeof(int8_t)
      + CTA_K * HEAD_DIM * sizeof(int8_t);
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("Sparse Sage SA2++ shared-memory request failed: ")
        + cudaGetErrorString(error));
  }

  dim3 grid((q_length + CTA_Q - 1) / CTA_Q, q_heads, batch);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  kernel<<<grid, block, smem, stream>>>(
      const_cast<int8_t *>(q), const_cast<int8_t *>(k),
      const_cast<int8_t *>(v), output, nullptr, const_cast<int32_t *>(lut),
      const_cast<int32_t *>(valid), const_cast<float *>(pv_threshold),
      const_cast<float *>(q_scale), const_cast<float *>(k_scale),
      const_cast<float *>(v_scale), q_length, kv_length, q_heads / kv_heads,
      q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
      v_stride_b, v_stride_h, v_stride_d, o_stride_b, o_stride_n, o_stride_h,
      scale);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("Sparse Sage SA2++ kernel launch failed: ")
        + cudaGetErrorString(error));
  }
}

} // namespace

void launch_h3_sparse_sage_sa2pp_sm89(
    const void *q, const void *k, const void *v, void *output,
    const void *lut, const void *valid, const void *pv_threshold,
    const void *q_scale, const void *k_scale, const void *v_scale,
    int batch, int q_length, int kv_length, int q_heads, int kv_heads,
    int head_dim, int q_stride_b, int q_stride_n, int q_stride_h,
    int k_stride_b, int k_stride_n, int k_stride_h,
    int v_stride_b, int v_stride_h, int v_stride_d,
    int o_stride_b, int o_stride_n, int o_stride_h,
    float scale, int output_dtype_code, cudaStream_t stream) {
  if (!q || !k || !v || !output || !lut || !valid || !pv_threshold
      || !q_scale || !k_scale || !v_scale) {
    throw std::runtime_error("Sparse Sage SA2++ received a null tensor pointer");
  }
  if (batch <= 0 || q_length <= 0 || kv_length <= 0 || q_heads <= 0
      || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim != HEAD_DIM) {
    throw std::runtime_error("Sparse Sage SA2++ received an unsupported shape");
  }

#define LAUNCH(DTYPE)                                                          \
  launch_impl(                                                                \
      static_cast<const int8_t *>(q), static_cast<const int8_t *>(k),         \
      static_cast<const int8_t *>(v), static_cast<DTYPE *>(output),           \
      static_cast<const int32_t *>(lut), static_cast<const int32_t *>(valid), \
      static_cast<const float *>(pv_threshold),                               \
      static_cast<const float *>(q_scale), static_cast<const float *>(k_scale), \
      static_cast<const float *>(v_scale), batch, q_length, kv_length,        \
      q_heads, kv_heads, q_stride_b, q_stride_n, q_stride_h, k_stride_b,      \
      k_stride_n, k_stride_h, v_stride_b, v_stride_h, v_stride_d,             \
      o_stride_b, o_stride_n, o_stride_h, scale, stream)

  if (output_dtype_code == 1) {
    LAUNCH(half);
  } else if (output_dtype_code == 2) {
    LAUNCH(nv_bfloat16);
  } else {
    throw std::runtime_error("Sparse Sage SA2++ output must be FP16 or BF16");
  }
#undef LAUNCH
}
