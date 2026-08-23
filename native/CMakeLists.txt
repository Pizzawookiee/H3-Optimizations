# SPDX-License-Identifier: Apache-2.0
#
# A plain shared library, not a Python extension. The whole point of the C ABI
# in h3_int8_attention_api.cu is that Python reaches it through ctypes, so
# nothing here links against Python and one binary serves every interpreter.
#
# No CUTLASS, no cuBLAS, no flash-attention: the vendored INT8 attention subset
# depends on nothing but the CUDA toolkit, which is what makes a single fat
# multi-architecture binary practical.

cmake_minimum_required(VERSION 3.26)
project(h3_int8_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED H3_CUDA_ARCHS)
    # Real SASS for the older parts and PTX on the newest, so hardware that
    # does not exist yet degrades to a JIT rather than failing to load. This
    # mirrors Kitchen's own arch policy; Windows toolkits often lack 90a.
    #
    # No sm_75. Turing reaches the INT8 tensor cores through a different path
    # -- larger fragments assembled from m8n8k16 rather than the native
    # m16n8k32 -- so "it compiled" exercises code the SM89 validation never
    # touches, and nobody has run it. It also costs ~9 MB in every shipped
    # binary. Ampere upward covers anyone realistically running a 56-head
    # video model; add it back only with a Turing card to test on.
    if(WIN32)
        set(H3_CUDA_ARCHS "80-real;89;120f")
    else()
        set(H3_CUDA_ARCHS "80-real;89;90a-real;120f")
    endif()
endif()
message(STATUS "H3 INT8 attention architectures: ${H3_CUDA_ARCHS}")
set(CMAKE_CUDA_ARCHITECTURES ${H3_CUDA_ARCHS})

add_library(h3_int8_attention SHARED
    src/h3_int8_attention_api.cu
    src/sage_attention/sage_attn_launcher.cu
    src/sage_attention/sage_attn_sparse_launcher.cu
    src/sage_attention/quant_qk_int8.cu
    src/sage_attention/quant_v_int8.cu
    src/sage_attention/v_staging.cu
)

target_include_directories(h3_int8_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sage_attention
)

target_compile_options(h3_int8_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT162_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>
)

# /EHs, deliberately NOT /EHsc. The trailing c in /EHsc means "extern \"C\"
# functions never throw", and the vendored launchers throw across exactly that
# boundary -- which is what MSVC keeps warning about as C4297. Compiled with
# /EHsc the compiler elides the unwind and the throw terminates the process
# instead of reaching the catch in h3_int8_attention_api.cu, so the guard that
# exists to keep exceptions away from ctypes is itself compiled away.
if(MSVC)
    target_compile_options(h3_int8_attention PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHs>
        $<$<COMPILE_LANGUAGE:CXX>:/EHs>
    )
    # CMake puts /EHsc in the default flags; strip it so ours is not overridden.
    string(REPLACE "/EHsc" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    
endif()

# Static CUDA runtime: one fewer thing for a user to have on their path.
set_target_properties(h3_int8_attention PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

if(WIN32)
    set_target_properties(h3_int8_attention PROPERTIES PREFIX "")
endif()

install(TARGETS h3_int8_attention LIBRARY DESTINATION . RUNTIME DESTINATION .)
