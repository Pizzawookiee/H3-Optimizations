// SPDX-License-Identifier: Apache-2.0
//
// A side-car C ABI for the two-phase V staging kernels.
//
// The same two entry points also exist in h3_int8_attention_api.cu, so a
// normal CMake rebuild puts them in the shipped library and this file is
// unnecessary. It exists because the experiment should not require rebuilding
// -- and therefore risking -- the 33 MB binary whose per-GPU self-test gates
// the whole sparse backend. v_staging.cu depends on nothing but
// dtype_dispatch.cuh and float_utils.cuh, so it compiles alone in seconds for
// one architecture, and the loader prefers the shipped library whenever that
// one already exports the symbols.
//
// Same conventions as the main ABI: every entry point is noexcept, returns 0
// on success, and leaves its message for h3_v_staging_last_error().

#include <cuda_runtime.h>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define H3V_API __declspec(dllexport)
#else
#define H3V_API __attribute__((visibility("default")))
#endif

void launch_v_amax_chunk(const void *v, void *amax, int B, int H, int rows,
                         int D, int64_t sb, int64_t sh, int64_t sn,
                         int input_dtype_code, cudaStream_t stream);

void launch_quant_v_chunk_into(const void *v, void *out, const void *scale,
                               int B, int H, int rows, int row_start, int D,
                               int padded_N, int64_t sb, int64_t sh, int64_t sn,
                               int input_dtype_code, cudaStream_t stream);

namespace {

thread_local std::string g_last_error;

void set_error(const char *what) {
  g_last_error.assign(what ? what : "unknown native error");
}

} // namespace

#define H3V_GUARD(BODY)                                                        \
  try {                                                                        \
    BODY;                                                                      \
    g_last_error.clear();                                                      \
    return 0;                                                                  \
  } catch (const std::exception &error) {                                      \
    set_error(error.what());                                                   \
    return 1;                                                                  \
  } catch (...) {                                                              \
    set_error("unknown native error");                                         \
    return 2;                                                                  \
  }

extern "C" {

H3V_API int h3_v_staging_abi_version() noexcept { return 1; }

H3V_API const char *h3_v_staging_last_error() noexcept {
  return g_last_error.empty() ? "" : g_last_error.c_str();
}

H3V_API int h3_int8_v_amax_chunk(const void *v, void *amax, int B, int H,
                                 int rows, int D, int64_t sb, int64_t sh,
                                 int64_t sn, int input_dtype_code,
                                 uintptr_t stream) noexcept {
  H3V_GUARD(launch_v_amax_chunk(v, amax, B, H, rows, D, sb, sh, sn,
                                input_dtype_code,
                                reinterpret_cast<cudaStream_t>(stream)))
}

H3V_API int h3_int8_quantize_v_chunk_into(const void *v, void *out,
                                          const void *scale, int B, int H,
                                          int rows, int row_start, int D,
                                          int padded_N, int64_t sb, int64_t sh,
                                          int64_t sn, int input_dtype_code,
                                          uintptr_t stream) noexcept {
  H3V_GUARD(launch_quant_v_chunk_into(v, out, scale, B, H, rows, row_start, D,
                                      padded_N, sb, sh, sn, input_dtype_code,
                                      reinterpret_cast<cudaStream_t>(stream)))
}

} // extern "C"
