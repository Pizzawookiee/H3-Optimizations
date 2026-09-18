/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * H3 grouped K/V producer. One ConvRot-INT8 GEMM produces a single K head
 * and matching V head. The epilogue writes V as BF16 while K is normalized,
 * RoPE-rotated, summarized, anchor-centered, Kitchen-rotated and quantized
 * directly into the final INT8 carrier. No BF16 K slab reaches global memory.
 */

#include <cuda_runtime.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

#include "float_utils.cuh"

namespace {
using namespace cute;

__forceinline__ __device__ void h3_convrot4(float *values) {
  const float x0 = values[0], x1 = values[1], x2 = values[2], x3 = values[3];
  const float a0 = x0 + x1, a1 = x0 - x1, a2 = x2 + x3, a3 = x2 - x3;
  values[0] = (a0 + a2) * 0.5f;
  values[1] = (a1 + a3) * 0.5f;
  values[2] = (a0 - a2) * 0.5f;
  values[3] = (a1 - a3) * 0.5f;
}

__forceinline__ __device__ void h3_convrot_sign128(float *values, int lane) {
  constexpr uint32_t signs_0 = 0x1035997bu;
  constexpr uint32_t signs_1 = 0x8087f5eeu;
  constexpr uint32_t signs_2 = 0xee2e4e1au;
  constexpr uint32_t signs_3 = 0x71132418u;
  const uint32_t signs = lane < 8 ? signs_0 : lane < 16 ? signs_1 : lane < 24 ? signs_2 : signs_3;
  const int shift = (lane & 7) * 4;
#pragma unroll
  for (int channel = 0; channel < 4; ++channel) {
    const uint32_t flip = ((signs >> (shift + channel)) & 1u) ^ 1u;
    values[channel] = __uint_as_float(__float_as_uint(values[channel]) ^ (flip << 31));
  }
}

__forceinline__ __device__ void h3_convrot128(float *values) {
  const int lane = threadIdx.x & 31;
  h3_convrot_sign128(values, lane);
  h3_convrot4(values);
#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int channel = 0; channel < 4; ++channel) {
      const float other = __shfl_xor_sync(0xffffffffu, values[channel], bit);
      values[channel] = (lane & bit) ? other - values[channel] : values[channel] + other;
    }
  }
#pragma unroll
  for (int channel = 0; channel < 4; ++channel)
    values[channel] *= 0.1767766952966369f;
}

template <typename Element>
__forceinline__ __device__ void normalize_rope4(
    const Element *tile, int local_row, int channel, const Element *norm,
    const Element *freqs, int rows, float epsilon, float *out) {
  float sum = 0.f;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const float value = local_row < rows
                            ? static_cast<float>(tile[local_row * 256 + channel + item])
                            : 0.f;
    out[item] = value;
    sum += value * value;
  }
  sum += __shfl_xor_sync(0xffffffffu, sum, 16);
  sum += __shfl_xor_sync(0xffffffffu, sum, 8);
  sum += __shfl_xor_sync(0xffffffffu, sum, 4);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  if (local_row >= rows) {
#pragma unroll
    for (int item = 0; item < 4; ++item) out[item] = 0.f;
    return;
  }
  const float rrms = rsqrtf(sum / 128.f + epsilon);
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    out[item] = static_cast<float>(Element(out[item] * rrms * static_cast<float>(norm[channel + item])));
  }

  const int lane = threadIdx.x & 31;
  const int pair_lane = lane < 12 ? lane + 12 : lane < 24 ? lane - 12 : lane;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const float other = __shfl_sync(0xffffffffu, out[item], pair_lane);
    if (lane < 24) {
      const float low = lane < 12 ? out[item] : other;
      const float high = lane < 12 ? other : out[item];
      const int pair = (lane % 12) * 4 + item;
      const Element *rotation = freqs + (int64_t)local_row * 48 * 4 + pair * 4;
      const int rotation_row = lane < 12 ? 0 : 2;
      const float first_rotation = static_cast<float>(rotation[rotation_row]);
      const float second_rotation = static_cast<float>(rotation[rotation_row + 1]);
      const float first = static_cast<float>(Element(low * first_rotation));
      const float second = static_cast<float>(Element(high * second_rotation));
      out[item] = static_cast<float>(Element(first + second));
    }
  }
}

template <class ThreadMap, class Element, int TBM, int TBN>
struct VisitorH3KVStore {
  struct Arguments {
    int8_t *k = nullptr;
    float *k_scale = nullptr;
    Element *summary = nullptr;
    Element *v = nullptr;
    const Element *norm = nullptr;
    const Element *freqs = nullptr;
    const Element *anchor = nullptr;
    const int *anchor_index = nullptr;
    int rows = 0;
    int full_rows = 0;
    int k_start = 0;
    int cta_k = 64;
    int full_k_length = 0;
    float epsilon = 0.f;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const &, Arguments const &args, void *) { return args; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &, Arguments const &) { return 0; }

  struct SharedStorage { alignas(16) Element tile[TBM * TBN]; };
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * cutlass::sizeof_bits<Element>::value;
  using VecType = cute::uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE VisitorH3KVStore() = default;
  CUTLASS_HOST_DEVICE VisitorH3KVStore(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params), tile(const_cast<Element *>(shared_storage.tile)) {}

  Params const *params_ptr;
  Element *tile;

  template <class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : cutlass::epilogue::threadblock::detail::EmptyCallbacks {
    CUTLASS_DEVICE Callbacks(RTensor &&tC_rAux, CTensor &&tC_cAux, ProblemShape problem_shape,
                             Params const *params_ptr, Element *tile, int tile_m, int tile_n)
        : tC_rAux(cute::forward<RTensor>(tC_rAux)), tC_cAux(cute::forward<CTensor>(tC_cAux)),
          problem_shape(problem_shape), params_ptr(params_ptr), tile(tile), tile_m(tile_m), tile_n(tile_n) {}

    RTensor tC_rAux; CTensor tC_cAux; ProblemShape problem_shape;
    Params const *params_ptr; Element *tile; int tile_m; int tile_n;

    CUTLASS_DEVICE void begin_step(int) { clear(tC_rAux); }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto visit(int, int, int, int frg_idx,
                              cutlass::Array<ElementAccumulator, FragmentSize> const &,
                              cutlass::Array<ElementInput, FragmentSize> const &frg_input) {
      using ConvertInput = cutlass::NumericArrayConverter<Element, ElementInput, FragmentSize,
          cutlass::FloatRoundStyle::round_to_nearest>;
      ConvertInput convert_input{};
      auto tC_rAux_frg = recast<cutlass::Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);
      return frg_input;
    }

    CUTLASS_DEVICE void end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
#pragma unroll
      for (int i = 0; i < size(src_v); ++i) {
        const bool guard = elem_less(coord_v(i), problem_shape);
        if (guard) {
          const int row = int(get<0>(coord_v(i))) - tile_m;
          const int column = int(get<1>(coord_v(i))) - tile_n;
          *reinterpret_cast<VecType *>(&tile[row * TBN + column]) = src_v(i);
        }
      }
    }

    CUTLASS_DEVICE void end_epilogue() {
      __syncthreads();

      // V is the second 128 columns. Each threadblock owns at most 128 rows.
      for (int index = threadIdx.x; index < 128 * 128; index += blockDim.x) {
        const int row = index / 128;
        const int channel = index & 127;
        const int local_row = tile_m + row;
        if (local_row < params_ptr->rows)
          params_ptr->v[(int64_t)local_row * 128 + channel] =
              tile[row * 256 + 128 + channel];
      }
      __syncthreads();

      const int lane = threadIdx.x & 31;
      const int warp = threadIdx.x >> 5;
      const int channel = lane * 4;
      const int blocks_in_tile = params_ptr->cta_k == 64 ? 2 : 1;
      float summary_sum[2][4] = {};

      if (threadIdx.x < 128) {
        const bool use_anchor =
            params_ptr->anchor_index && params_ptr->anchor_index[0] >= 0;
        float anchor4[4] = {};
        if (use_anchor) {
#pragma unroll
          for (int item = 0; item < 4; ++item)
            anchor4[item] = static_cast<float>(params_ptr->anchor[channel + item]);
        }

#pragma unroll
        for (int block = 0; block < 2; ++block) {
          if (block >= blocks_in_tile) break;
          const int block_row0 = block * params_ptr->cta_k;
          const int rows_per_warp = params_ptr->cta_k / 4;
          const int iterations = rows_per_warp / 2;
          float values[32 * 4];
          float maximum = 0.f;

#pragma unroll
          for (int i = 0; i < 32 * 4; ++i) values[i] = 0.f;

          for (int j = 0; j < iterations; ++j) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
              const int row_in_block = j * 8 + warp * 2 + p;
              const int row_in_tile = block_row0 + row_in_block;
              const int local_row = tile_m + row_in_tile;
              const int vi = (j * 2 + p) * 4;
              float row_values[4];
              normalize_rope4(
                  tile, row_in_tile, channel, params_ptr->norm,
                  params_ptr->freqs + (int64_t)tile_m * 48 * 4,
                  params_ptr->rows - tile_m, params_ptr->epsilon, row_values);
#pragma unroll
              for (int item = 0; item < 4; ++item) {
                summary_sum[block][item] += row_values[item];
                row_values[item] -= anchor4[item];
                values[vi + item] = row_values[item];
              }
              if (local_row < params_ptr->rows) {
                if (params_ptr->full_k_length > 256)
                  h3_convrot128(&values[vi]);
                else
                  h3_convrot4(&values[vi]);
#pragma unroll
                for (int item = 0; item < 4; ++item)
                  maximum = fmaxf(maximum, fabsf(values[vi + item]));
              }
            }
          }

          maximum = comfy::warp_reduce_fmax(maximum);
          const float scale = maximum / 127.f + 1e-7f;
          const float inverse_scale = 1.f / scale;
          const int global_block =
              (params_ptr->k_start + tile_m + block_row0) / params_ptr->cta_k;
          if (lane == 0 &&
              params_ptr->k_start + tile_m + block_row0 < params_ptr->full_rows)
            params_ptr->k_scale[global_block * 4 + warp] = scale;

          for (int j = 0; j < iterations; ++j) {
#pragma unroll
            for (int p = 0; p < 2; ++p) {
              const int row_in_block = j * 8 + warp * 2 + p;
              const int local_row = tile_m + block_row0 + row_in_block;
              if (local_row < params_ptr->rows) {
                const int vi = (j * 2 + p) * 4;
                const int global_row = params_ptr->k_start + local_row;
                int8_t *output =
                    params_ptr->k + (int64_t)global_row * 128 + channel;
                comfy::store4_i8(
                    output,
                    comfy::quant_int8_rcp(values[vi], inverse_scale),
                    comfy::quant_int8_rcp(values[vi + 1], inverse_scale),
                    comfy::quant_int8_rcp(values[vi + 2], inverse_scale),
                    comfy::quant_int8_rcp(values[vi + 3], inverse_scale));
              }
            }
          }
        }
      }

      // All GEMM threads must reach block barriers, even though only the first
      // 128 threads own the K head.
      __syncthreads();
      float *partials = reinterpret_cast<float *>(tile);
      if (threadIdx.x < 128) {
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          if (block < blocks_in_tile) {
#pragma unroll
            for (int item = 0; item < 4; ++item)
              partials[((block * 4 + warp) * 128) + channel + item] =
                  summary_sum[block][item];
          }
        }
      }
      __syncthreads();

      if (threadIdx.x < 128) {
        for (int index = threadIdx.x; index < blocks_in_tile * 128;
             index += 128) {
          const int block = index / 128;
          const int summary_channel = index & 127;
          const int local_start = tile_m + block * params_ptr->cta_k;
          const int remaining = params_ptr->rows - local_start;
          const int valid_rows =
              remaining < params_ptr->cta_k ? remaining : params_ptr->cta_k;
          if (valid_rows > 0) {
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < 4; ++w)
              sum += partials[((block * 4 + w) * 128) + summary_channel];
            const int global_block =
                (params_ptr->k_start + local_start) / params_ptr->cta_k;
            params_ptr->summary[(int64_t)global_block * 128 + summary_channel] =
                Element(sum / valid_rows);
          }
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto get_callbacks(cutlass::gemm::GemmCoord threadblock_tile_offset,
                                    int thread_idx, ProblemShape problem_shape) {
    const int64_t m = get<0>(problem_shape), n = get<1>(problem_shape);
    auto dummy = make_tensor(make_gmem_ptr(static_cast<Element *>(nullptr)), problem_shape,
                             cute::Stride<int64_t, _1, int64_t>{n, _1{}, m * n});
    auto partitioned = group_modes<3, 6>(ThreadMap::partition(dummy, thread_idx, threadblock_tile_offset));
    auto tC_rAux = make_tensor_like(take<0, 3>(recast<VecType>(partitioned)));
    auto coordinates = make_identity_tensor(dummy.shape());
    auto tC_cAux = outer_partition(group_modes<3, 6>(ThreadMap::partition(coordinates, thread_idx, threadblock_tile_offset)),
                                   Shape<Int<VecLength>>{}, (_0{}));
    return Callbacks<decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr, tile,
        threadblock_tile_offset.m() * TBM, threadblock_tile_offset.n() * TBN);
  }
};

struct H3FusedKVGemm {
  using ElementA = int8_t; using ElementB = int8_t; using ElementC = cutlass::bfloat16_t;
  using ElementAcc = int32_t; using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor; using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using TB = cutlass::gemm::GemmShape<128, 256, 64>;
  using Warp = cutlass::gemm::GemmShape<64, 64, 64>;
  using Inst = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int Align = 16, AlignC = 8, EVTStages = 1;
  using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, Warp, ElementC, AlignC, EVTStages>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<ThreadMap, ElementCompute, cute::Stride<_1, _0, int32_t>>;
  using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<ThreadMap, ElementCompute, cute::Stride<_0, _1, int32_t>>;
  using Mul0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Mul0, Accum, XScale>;
  using Mul1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementC, ElementCompute,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT1 = cutlass::epilogue::threadblock::Sm80EVT<Mul1, EVT0, WScale>;
  using StoreKV = VisitorH3KVStore<ThreadMap, ElementC, 128, 256>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<StoreKV, EVT1>;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA, LayoutA, cutlass::ComplexTransform::kNone, Align,
      ElementB, LayoutB, cutlass::ComplexTransform::kNone, Align,
      ElementC, LayoutC, AlignC, ElementAcc, ElementCompute,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, Warp, Inst, EVTD,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
      cutlass::arch::OpMultiplyAddSaturate, EVTStages>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static bool run(const int8_t *a, const int8_t *b, const float *x_scale,
                  const float *weight_scale, const ElementC *norm,
                  const ElementC *freqs, const ElementC *anchor,
                  const int *anchor_index, int8_t *k_out, float *k_scale,
                  ElementC *summary, ElementC *v_out, int rows, int hidden,
                  int full_rows, int k_start, int cta_k, int full_k_length,
                  float epsilon, cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem(rows, 256, hidden);
    typename EVTD::Arguments callbacks{
        {{{}, {const_cast<float *>(x_scale), 0.f, {_1{}, _0{}, rows}}, {}},
         {const_cast<float *>(weight_scale), 0.f, {_0{}, _1{}, 256}}, {}},
        {k_out, k_scale, summary, v_out, norm, freqs, anchor, anchor_index,
         rows, full_rows, k_start, cta_k, full_k_length, epsilon}};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, problem, 1, callbacks,
        const_cast<int8_t *>(a), const_cast<int8_t *>(b), nullptr, nullptr,
        (int64_t)rows * hidden, (int64_t)256 * hidden, 0, 0, hidden, hidden, 0, 0);
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    const size_t workspace_size = Gemm::get_workspace_size(args);
    if (workspace_size != 0) return false;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
    return gemm(stream) == cutlass::Status::kSuccess;
  }
};
} // namespace

bool launch_h3_fused_kv_cutlass(
    const void *a, const void *b, const void *x_scale, const void *weight_scale,
    const void *norm, const void *freqs, const void *anchor,
    const void *anchor_index, void *k_out, void *k_scale, void *summary,
    void *v_out, int64_t rows, int64_t hidden, int64_t full_rows,
    int64_t k_start, int cta_k, int full_k_length, float epsilon,
    cudaStream_t stream) {
  if (rows == 0) return true;
  if (cta_k != 64 && cta_k != 128) return false;
  return H3FusedKVGemm::run(
      static_cast<const int8_t *>(a), static_cast<const int8_t *>(b),
      static_cast<const float *>(x_scale), static_cast<const float *>(weight_scale),
      static_cast<const cutlass::bfloat16_t *>(norm),
      static_cast<const cutlass::bfloat16_t *>(freqs),
      static_cast<const cutlass::bfloat16_t *>(anchor),
      static_cast<const int *>(anchor_index), static_cast<int8_t *>(k_out),
      static_cast<float *>(k_scale), static_cast<cutlass::bfloat16_t *>(summary),
      static_cast<cutlass::bfloat16_t *>(v_out), static_cast<int>(rows),
      static_cast<int>(hidden), static_cast<int>(full_rows), static_cast<int>(k_start),
      cta_k, full_k_length, epsilon, stream);
}
