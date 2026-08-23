# Vendored INT8 attention

The pack ships its own compiled INT8 attention instead of depending on a
comfy-kitchen release.

## Why this exists

ComfyUI pins `comfy-kitchen==0.2.31` in its requirements. Anything that lives
only in a later Kitchen, or in a fork, reaches nobody — which is exactly what
happened to the chunked QKV producer: fully integrated on this side, calling a
Kitchen API that was never released, so every install silently fell back to
standard projection at roughly half the speed.

Two pieces are needed and neither is in 0.2.31:

- **the external INT8 attention producer**, which lets a chunked QKV
  projection quantize each chunk as it is produced rather than materializing
  the whole BF16 sequence;
- **block-sparse KV traversal**, which walks a routed subset of the KV tiles.

## Provenance

`src/sage_attention/` and the two headers beside it are vendored from Comfy
Kitchen (Apache-2.0, see `LICENSE` and `NOTICE.upstream`), which derives them
in turn from SageAttention. The exact commit is recorded in `src/PROVENANCE`.

Only one change was made to the vendored sources, and it is marked in each
file with a `VENDORING CHANGE` comment: the launchers lost their `extern "C"`
linkage. Upstream they are reached through nanobind; here they are internal to
this library and only `h3_int8_attention_api.cu` calls them. C linkage bought
nothing and cost correctness — under MSVC's `/EHsc` the trailing `c` means
*extern "C" functions never throw*, so the compiler elides the unwind and a
throw terminates the process instead of reaching the catch in the API layer.
Upstream warns about this as C4297 on every build; here it was fatal until the
linkage changed.

## The API boundary

`src/h3_int8_attention_api.cu` is the stable surface. Every entry point is
`noexcept`, returns `0` on success or non-zero on failure, and leaves a message
for `h3_int8_last_error()`. `h3_optimizations/native/loader.py` turns that into
an ordinary Python exception.

It is plain C — pointers, ints, a stream handle — loaded through `ctypes`. No
nanobind, no DLPack, and therefore no Python ABI dimension: one binary per OS
and architecture set serves every interpreter, rather than separate cp310,
cp311 and abi3 builds.

## Building

```
cmake -S native -B native/build
cmake --build native/build --config Release
```

The loader finds `native/build/Release/`, `native/build/`, or `native/lib/`,
and `H3_INT8_ATTENTION_LIBRARY` overrides all of them.

Architectures default to `75-real;80-real;89;120f` on Windows and add
`90a-real` on Linux: real SASS for the older parts and PTX on the newest, so
hardware that does not exist yet degrades to a JIT rather than failing to load.
Override with `-DH3_CUDA_ARCHS=89` to build just one while iterating.

There is no CUTLASS, cuBLAS or flash-attention dependency — the vendored subset
needs only the CUDA toolkit, which is what makes a single fat multi-architecture
binary practical.
