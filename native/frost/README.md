# FROST BF16 SM89 artifact

This directory contains the packaged CUDA cubin used by the explicit
`FROST BF16 (SM89)` sparse-attention backend. It is a project-built derivative
of NVIDIA's Apache-2.0 `prefill_f16_sm80.py` FROST template, not a wheel-provided
kernel.

The specialization accepts BF16 H3 Q in `[1, 56, SQ, 128]`, BF16 K/V in
`[1, 56, SKV, 128]`, a full-width absolute int32 block route with int32 row
counts, and writes sequence-major BF16 output. `SQ` and `SKV` may differ, so a
bounded Q slab can attend the global K/V carrier. Query and KV tiles are both
64 tokens. Only SM89 is accepted by runtime preflight.

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

The routed derivative retains NVIDIA's two-stage `cp.async` schedule. Current V
and the next selected K tile are addressed independently through the absolute
route, so arbitrary sparse selections do not rely on contiguous pointer state.

## Validation

The packaged 56-head artifact passed CUDA Driver parity on an RTX 4070 / SM89
for Q lengths 64, 65, and 129 against 193-row K/V, using both dense and
noncontiguous absolute routes. Relative L2 error was at most 0.00258 and all
outputs were finite. The streamed consumer also passed a three-slab
`64 + 64 + 1` execution against global 129-row K/V with relative L2 0.00258.

The pipelined artifact was bit-identical to the synchronous baseline for dense,
noncontiguous, tail, single-entry, and unequal Q/KV sequence cases. At the
kernel boundary on the same RTX 4070, a 4096-row Q slab against 54,006-row K/V
at 30.3% tile density improved from 52.38 ms to 43.00 ms median (1.218x).
