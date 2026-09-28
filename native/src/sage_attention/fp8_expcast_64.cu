// SPDX-License-Identifier: Apache-2.0

#include "qk_int_sv_i8_sparse_cuda.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

constexpr int CTA_Q = 64;
constexpr int CTA_K = 64;
constexpr int WARP_Q = 16;
constexpr int WARP_K = 64;
constexpr int HEAD_DIM = 128;

template <typename DTypeOut, bool EXPCAST, bool PRE_FP8_DENOM, bool V_SMOOTH>
void launch_impl(
    const int8_t *q, const int8_t *k, const int8_t *v, DTypeOut *output,
    const float *q_scale, const float *k_scale, const float *v_scale,
    const float *v_mean,
    const int32_t *lut, const int32_t *valid, const int32_t *kv_tile_len,
    int lut_stride, int batch, int q_length, int kv_length, int q_heads,
    int kv_heads, int q_stride_b, int q_stride_n, int q_stride_h,
    int k_stride_b, int k_stride_n, int k_stride_h, int v_stride_b,
    int v_stride_h, int v_stride_d, int o_stride_b, int o_stride_n,
    int o_stride_h, int qs_stride_b, int qs_stride_h, float scale,
    cudaStream_t stream) {
  auto kernel = qk_int_sv_i8_sparse_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float,
      false, DTypeOut, ComputeUnit::kCudaCore, MaskMode::kNone, false,
      !V_SMOOTH,
      false, false, true, true, true, EXPCAST, PRE_FP8_DENOM, V_SMOOTH>;
  constexpr size_t smem = CTA_Q * HEAD_DIM * sizeof(int8_t)
      + CTA_K * HEAD_DIM * sizeof(int8_t)
      + CTA_K * HEAD_DIM * sizeof(int8_t);
  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("FP8 64x64 shared-memory request failed: ")
        + cudaGetErrorString(error));
  }

  dim3 grid(div_ceil(q_length, CTA_Q), q_heads, batch);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));
  kernel<<<grid, block, smem, stream>>>(
      const_cast<int8_t *>(q), const_cast<int8_t *>(k),
      const_cast<int8_t *>(v), output, nullptr, const_cast<float *>(q_scale),
      const_cast<float *>(k_scale), const_cast<float *>(v_scale),
      const_cast<float *>(v_mean),
      nullptr, 0, 0, 0, 0, 0, q_length, kv_length, q_heads / kv_heads,
      q_stride_b, q_stride_n, q_stride_h, k_stride_b, k_stride_n, k_stride_h,
      v_stride_b, v_stride_h, v_stride_d, o_stride_b, o_stride_n, o_stride_h,
      qs_stride_b, qs_stride_h, scale, lut, valid,
      static_cast<uint32_t>(lut_stride), kv_tile_len);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("FP8 64x64 kernel launch failed: ")
        + cudaGetErrorString(error));
  }
}

} // namespace

void launch_h3_fp8_attention_64(
    const void *q, const void *k, const void *v, void *output,
    const void *q_scale, const void *k_scale, const void *v_scale,
    const void *v_mean,
    const void *lut, const void *valid, const void *kv_tile_len,
    int lut_stride, int batch, int q_length, int kv_length, int q_heads,
    int kv_heads, int head_dim, int q_stride_b, int q_stride_n,
    int q_stride_h, int k_stride_b, int k_stride_n, int k_stride_h,
    int v_stride_b, int v_stride_h, int v_stride_d, int o_stride_b,
    int o_stride_n, int o_stride_h, int qs_stride_b, int qs_stride_h,
    float scale, int output_dtype_code, bool expcast, bool pre_fp8_denom,
    bool v_smooth,
    cudaStream_t stream) {
  if (!q || !k || !v || !output || !q_scale || !k_scale || !v_scale
      || !lut || !valid || !kv_tile_len) {
    throw std::runtime_error("FP8 64x64 attention received a null pointer");
  }
  if (batch <= 0 || q_length <= 0 || kv_length <= 0 || q_heads <= 0
      || kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim != HEAD_DIM
      || kv_length % CTA_K != 0) {
    throw std::runtime_error("FP8 64x64 attention received unsupported geometry");
  }
  if (v_smooth && !v_mean) {
    throw std::runtime_error("V-Smooth attention received a null mean pointer");
  }
  if (pre_fp8_denom && (expcast || v_smooth)) {
    throw std::runtime_error(
        "pre-FP8 denominator supports only ordinary FP8 probabilities");
  }

#define H3_LAUNCH(DTYPE, EXPCAST, PREDENOM, VSMOOTH)                          \
  launch_impl<DTYPE, EXPCAST, PREDENOM, VSMOOTH>(                            \
      static_cast<const int8_t *>(q), static_cast<const int8_t *>(k),        \
      static_cast<const int8_t *>(v), static_cast<DTYPE *>(output),          \
      static_cast<const float *>(q_scale),                                   \
      static_cast<const float *>(k_scale),                                   \
      static_cast<const float *>(v_scale),                                   \
      static_cast<const float *>(v_mean),                                    \
      static_cast<const int32_t *>(lut),                                     \
      static_cast<const int32_t *>(valid),                                   \
      static_cast<const int32_t *>(kv_tile_len), lut_stride, batch,          \
      q_length, kv_length, q_heads, kv_heads, q_stride_b, q_stride_n,        \
      q_stride_h, k_stride_b, k_stride_n, k_stride_h, v_stride_b,            \
      v_stride_h, v_stride_d, o_stride_b, o_stride_n, o_stride_h,            \
      qs_stride_b, qs_stride_h, scale, stream)

  if (output_dtype_code == 1) {
    if (expcast) {
      if (v_smooth) H3_LAUNCH(half, true, false, true);
      else H3_LAUNCH(half, true, false, false);
    } else if (pre_fp8_denom) {
      H3_LAUNCH(half, false, true, false);
    } else {
      if (v_smooth) H3_LAUNCH(half, false, false, true);
      else H3_LAUNCH(half, false, false, false);
    }
  } else if (output_dtype_code == 2) {
    if (expcast) {
      if (v_smooth) H3_LAUNCH(nv_bfloat16, true, false, true);
      else H3_LAUNCH(nv_bfloat16, true, false, false);
    } else if (pre_fp8_denom) {
      H3_LAUNCH(nv_bfloat16, false, true, false);
    } else {
      if (v_smooth) H3_LAUNCH(nv_bfloat16, false, false, true);
      else H3_LAUNCH(nv_bfloat16, false, false, false);
    }
  } else {
    throw std::runtime_error("FP8 64x64 output must be FP16 or BF16");
  }
#undef H3_LAUNCH
}
