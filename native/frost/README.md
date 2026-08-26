# FROST BF16 SM89 artifact

This directory contains the packaged CUDA cubin used by the explicit
`FROST BF16 (SM89)` sparse-attention backend. It is a project-built derivative
of NVIDIA's Apache-2.0 `prefill_f16_sm80.py` FROST template, not a wheel-provided
kernel.

The specialization accepts BF16 H3 Q/K/V tensors in `[1, 42, S, 128]`, a
full-width absolute int32 block route with int32 row counts, and writes
sequence-major BF16 output. Query and KV tiles are both 64 tokens.
Only SM89 is accepted by runtime preflight.

## Rebuild

1. Check out `NVIDIA/cudnn-frontend` at
   `ae8705effeea3804585b6aca554beaca1a76a3da`.
2. Apply `frost_h3.patch` at the checkout root.
3. Build the container from this directory with
   `docker build -t h3-frost-sm89 .`.
4. Mount the patched checkout at `/work/cudnn-frontend`, this directory at
   `/work/h3-frost`, and an output directory at `/work/artifacts`.
5. Run the compiler without a GPU:
   `docker run --rm -v /path/to/cudnn-frontend:/work/cudnn-frontend -v
   /path/to/this/directory:/work/h3-frost -v /path/to/artifacts:/work/artifacts
   h3-frost-sm89 python /work/h3-frost/compile_sm89.py`.
6. Copy the emitted cubin to `h3_frost_bf16_sm89.cubin`, confirm its entry
   symbol matches `h3_frost_bf16_sm89.symbol`, and update the SHA256 in
   `PROVENANCE` if the toolchain intentionally changed.

The CUDA Driver ABI parameter order is:

`q, k, v, output, route, counts, seven null optional-feature pointers,
q_tiles, kv_tiles, n_kv_tiles, softmax_scale_log2, sequence_q, sequence_kv,
head_dim, band, inverse_softmax_scale`.

The routed derivative drains each matching K/V tile into shared memory before
computing it. NVIDIA's original cross-iteration prefetch assumes a contiguous
KV range and is not used by this sparse specialization.
