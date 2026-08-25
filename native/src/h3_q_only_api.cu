// SPDX-License-Identifier: Apache-2.0
// Benchmark-only C ABI for the Q-only parity experiment.

#include <cuda_runtime.h>
#include <cstdint>
#include <exception>
#include <string>

#if defined(_WIN32)
#define H3Q_API __declspec(dllexport)
#else
#define H3Q_API __attribute__((visibility("default")))
#endif

void launch_quant_q_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, int B, int H_q, int Lq, int C,
    int full_Lk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int input_dtype_code, cudaStream_t stream);

namespace {
thread_local std::string g_last_error;
}

#define H3Q_GUARD(BODY)                                                        \
  try {                                                                        \
    BODY;                                                                      \
    g_last_error.clear();                                                      \
    return 0;                                                                  \
  } catch (const std::exception &error) {                                      \
    g_last_error.assign(error.what());                                         \
    return 1;                                                                  \
  } catch (...) {                                                              \
    g_last_error.assign("unknown native error");                              \
    return 2;                                                                  \
  }

extern "C" {

H3Q_API int h3_q_only_abi_version() noexcept { return 1; }

H3Q_API const char *h3_q_only_last_error() noexcept {
  return g_last_error.empty() ? "" : g_last_error.c_str();
}

H3Q_API int h3_int8_quantize_q_only(
    const void *q, void *q_int8, void *q_scale, int B, int H_q, int Lq, int C,
    int full_Lk, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int input_dtype_code, uintptr_t stream) noexcept {
  H3Q_GUARD(launch_quant_q_per_thread_int8(
      q, q_int8, q_scale, B, H_q, Lq, C, full_Lk, q_stride_b, q_stride_h,
      q_stride_n, input_dtype_code, reinterpret_cast<cudaStream_t>(stream)))
}

} // extern "C"
