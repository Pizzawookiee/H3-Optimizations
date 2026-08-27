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
- **composable sparse softmax state**, which lets the same native traversal
  return its base-2 row normalizer for a separate rejected-tile residual.

## Provenance

`src/sage_attention/` and the two headers beside it are vendored from Comfy
Kitchen (Apache-2.0, see `LICENSE` and `NOTICE.upstream`), which derives them
in turn from SageAttention. The exact commit is recorded in `src/PROVENANCE`.

The launcher change is marked with a `VENDORING CHANGE` comment: the launchers
lost their `extern "C"` linkage. Upstream they are reached through nanobind;
here they are internal to
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

ABI 3 adds explicit sparse Q geometry and Q-scale strides. It exposes exact
128Q x 64KV and 64Q x 64KV traversal without changing Kitchen's Q128
quantization carrier. ABI 2 added `h3_int8_sparse_attention_lse`; that output
remains one FP32 base-2 log-sum-exp value per query row. The packaged Windows
DLL is build-versioned so a running Comfy process can keep an older mapped DLL
until the next normal restart.

## Building

```
cmake -S native -B native/build -G Ninja
cmake --build native/build --config Release
```

On Windows, run those commands from an x64 Visual Studio developer shell. The
Visual Studio CUDA MSBuild targets hardcode `/EHsc` after project flags, so the
Visual Studio generator is rejected; Ninja preserves the required `/EHs`
exception boundary described above.

The loader first uses the platform binary committed under `native/bin/`, then
checks local `native/lib/`, `native/build/`, and `native/build/Release/` paths.

To point it somewhere else, write that path into `native/library_path.txt`;
the first non-blank, non-`#` line wins and is searched ahead of everything
above. This replaces the old `H3_INT8_ATTENTION_LIBRARY` environment variable.
The Registry's automated package scanner reports any runtime `os.environ` read
as environment manipulation, and a flagged release leaves ComfyUI-Manager
pinned to the previous approved version, so the override became a file the
pack owns. It is ignored by both Git and `.comfyignore`, so it never exists in
a published install.

Architectures default to
`75-real;80-real;89-real;120f-real;89-virtual` on Windows and add `90a-real`
on Linux. Every shipped target has real SASS, while one `compute_89` PTX
payload provides the forward-compatible fallback without duplicating the
family-limited `compute_120f` PTX. Override with `-DH3_CUDA_ARCHS=89` to build
just one while iterating.

SM75 composes the Ampere-shaped INT8 MMA fragments from Turing's m8n8k16
instructions and uses synchronous shared-memory copies where newer GPUs use
`cp.async`. The packaged path is still accepted only after its per-GPU dense
and sparse numerical self-test passes.

Linux release binaries are built in the CUDA 13 Ubuntu 22.04 image with GCC 11
and pinned CMake 3.28.3. The packaged library requires no newer than
`GLIBCXX_3.4.21` and keeps its existing `GLIBC_2.34` floor. The shipping test
rejects a binary built against a newer libstdc++ ABI. CUDA fatbins use size
compression, unused ELF sections are discarded, and the final shared library
is stripped while retaining its exported C ABI.

There is no CUTLASS, cuBLAS or flash-attention dependency — the vendored subset
needs only the CUDA toolkit, which is what makes a single fat multi-architecture
binary practical.
