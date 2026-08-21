# Chunked Comfy Kitchen QKV cutover plan

## Decision

Use Comfy Kitchen for dense INT8 attention and for all dense carrier semantics, but
produce those carriers from H3 in bounded sequence chunks.

```text
hidden activation
  -> 4K-row Kitchen ConvRot QKV projections
  -> Kitchen RMSNorm + position-correct RoPE per Q/K chunk
  -> Kitchen Q/K quantization into final carrier slices
  -> one retained full BF16 V buffer
  -> Kitchen global V quantization
  -> Kitchen prequantized INT8 attention
  -> normal H3 output projection
```

This replaces the package-owned dense Sage stack after it passes the complete cutover
benchmark. It does not replace Sparse Sage. Sparse attention keeps its standard QKV path
and its optional native-carrier fused QKV producer.

The durable dense fallback is upstream H3 QKV plus the attention backend selected by
ComfyUI. The legacy dense Sage path is only a temporary benchmark reference and must be
deleted after acceptance.

## Why this is the target

The measured 54,006-token H3 block-0 preparation results were:

| Path | Median | Peak allocation delta |
| --- | ---: | ---: |
| Full Kitchen QKV + RMSNorm/RoPE + retained V | 188.130 ms | 2.885 GiB |
| Kitchen QKV in 4,096-row chunks | 99.822 ms | 1.071 GiB |
| Legacy fused dense-Sage carrier producer | 214.948 ms | 1.718 GiB |

The full and 4K Kitchen preparation paths produced exactly equal Q, K, and V. Kernel
tracing confirmed that both used the intended Kitchen ConvRot projection and
RMSNorm/RoPE kernels. The 4K result is therefore the initial policy, not an exposed UI
setting or a claim that 4K is optimal on every GPU.

The completed equal-boundary benchmark then measured:

| Path | Median | Peak allocation delta |
| --- | ---: | ---: |
| Full Kitchen QKV -> Kitchen INT8 attention | 880.773 ms | 3.251 GiB |
| 4K chunked Kitchen QKV -> Kitchen INT8 attention | 772.291 ms | 1.808 GiB |
| Legacy fused QKV -> dense Sage | 883.216 ms | 2.530 GiB |

The chunked path was bitwise-identical to full Kitchen for every Q/K/V carrier tensor
and for the complete attention output. It was 12.6 percent faster than legacy fused
Sage and reduced peak allocation by another 0.722 GiB. This clears the exact-boundary
memory, speed, and Kitchen-reference parity thresholds on the RTX 4070. Legacy Sage
differed from Kitchen by max absolute error 0.1171875 and relative RMSE 0.03470 because
the two paths use different V carriers. Fixed-workflow validation remains separate
acceptance work.

`benchmarks/bench_h3_block.py` loads every actual block-0 parameter and runs the production
2,048-row bounded ConvRot MLP forward. At 4,353 rows, which exercises a full 4K QKV
chunk plus a final partial chunk, it measured 39.478 ms for full Kitchen, 40.056 ms for
chunked Kitchen, and 37.475 ms for legacy fused Sage. Chunked Kitchen remained bitwise
identical to full Kitchen through the complete residual, modulation, attention, and MLP
block. The fixed workflow remains the final quality check.

`benchmarks/prepare_quality_workflows.py` now prepares that final check without changing
the user's source workflow. It removes intentionally retired serialized widgets from the
two current H3 nodes, bypasses Sparse Attention, substitutes an explicitly selected local
reference image, and emits matched full-Kitchen (`fused_qkv=off`) and chunked-Kitchen
(`fused_qkv=auto`) UI workflows plus a source-hashed manifest. The prepared local pair
holds the ref2va checkpoint, conditioning, seed `1056061618320794`, beta scheduler, 20
steps, 10-second duration, 1184 x 896 resolution, and bounded MLP settings constant. Its
execution remains pending the supported Kitchen API and separate full-workflow
authorization.

## Current implementation checkpoint

Completed in the development worktrees:

- isolated Kitchen producer ABI v1 and native CUDA bindings;
- the complete Kitchen INT8-attention suite on CUDA: 75 passed and two HIP-only tests
  skipped, including exact external-producer parity for unmasked CTA-K 64,
  unmasked CTA-K 128, and masked CTA-K 64;
- H3 4K chunk integration with public Kitchen calls only;
- explicit Comfy attention override composition in both application orders;
- sparse fused-QKV selection by complete carrier contract, including matching SM120;
- CPU/static pack suite and complete 54,006-token three-path benchmark;
- source-hashed, current-schema full-versus-chunked fixed-workflow pair prepared offline.

The Kitchen patch remains uncommitted on the isolated
`chunked-int8-attention-producer` branch so it can be reviewed without mutating an
upstream repository. It is based on current upstream HEAD
`ff83be31dd7f397975a0b8f2c0e86cd50ad78dd5`; no related upstream PR or existing
`Zironic/comfy-kitchen` fork is available through the authenticated GitHub connection.
Publishing a draft therefore still requires an explicitly authorized fork/write route.

Still required before legacy deletion:

- establish the producer API as a supported upstream Kitchen contract;
- run the fixed-workflow quality check;
- re-run final CPU/static and GPU acceptance after deleting the legacy dense stack.

The retained profiler trace proves label reachability: full and chunked Kitchen both
reach ConvRot activation quantization, CUTLASS INT8 GEMM, Kitchen RMSNorm/RoPE,
Kitchen Q/K/V packing, and Kitchen INT8 attention; the legacy label reaches
`_dense_fused_qkv_kernel`, Sage FP8-V preparation, and the dense Sage attention kernel.

| Phase | State |
| --- | --- |
| Fair legacy reference | Complete |
| Kitchen producer API | Prototype and parity complete; upstream support pending |
| Chunked H3 integration | Complete in the development worktree |
| SM89 policy removal | Sparse contract complete; legacy dense deletion pending |
| CPU/static acceptance | Complete: 89 tests pass with CUDA hidden |
| GPU exact boundary and real block | Complete on RTX 4070; fixed workflow prepared offline and execution pending |
| Legacy dense deletion | Pending upstream Kitchen support and workflow acceptance |

## Ownership after cutover

```text
ComfyUI / Comfy Kitchen owns
  ConvRot INT8 linear kernels and device dispatch
  Q/K RMSNorm + RoPE
  Q/K/V carrier geometry and scale semantics
  K-anchor sampling and selection
  carrier allocation, validation, and finalization
  dense INT8 attention kernels
  CUDA/HIP architecture support and wheel distribution

H3-Optimizations owns
  bounded H3 sequence scheduling and temporary lifetimes
  the narrow main-block integration
  bounded MLP execution
  video-only sparse routing
  Sparse Sage dispatch
  optional native-carrier sparse fused QKV
  a temporary source-signature-gated upstream V-layout shim
```

H3 production code must not call Kitchen private `_C` symbols, reproduce Kitchen's
anchor algorithm, infer carrier geometry from compute capability, or construct a
Kitchen carrier directly.

## Compatibility contracts

### Dense Kitchen producer

Chunked dense QKV is eligible only when Kitchen's public producer contract confirms all
of the following:

- supported producer ABI/version;
- compatible device, dtype, head dimension, mask, GQA, and sequence geometry;
- compatible ConvRot-256 TensorWise-INT8 QKV weight layout;
- Q/K tile geometry, carrier strides, scale layouts, padding, and valid-length rules;
- Kitchen-owned K-anchor sample positions and anchor selection;
- chunk-addressable Q/K carrier writes with absolute sequence offsets;
- global V quantization and validated carrier finalization;
- compatibility with `int8_attention_from_prequantized(...)`.

If any field or callable is missing, `auto` quietly uses upstream QKV. A future explicit
required mode, if added, must fail clearly instead of changing math or falling back to
legacy dense Sage.

The initial producer implementation is CUDA-only. HIP remains on standard QKV until
Kitchen advertises and validates the same producer-level contract for HIP.

### Sparse fused producer

Sparse fused QKV is selected through a complete producer/consumer match:

```text
active SparseSageKernelSpec
  device + Q tile + KV tile
  Q/K carrier and scale layouts
  V format and accumulator
  summary format
  exact required callables
        |
        v
sparse fused-QKV provider accepts or declines the whole contract
```

A matching non-SM89 architecture is eligible. A kernel that merely compiles is not.
SM120 may use the fused producer when its exported 128Q x 64KV contract matches.
Hopper's 64Q x 128KV contract declines until a matching producer exists. Decline means
standard sparse QKV, never dense attention.

### No SM89 policy gates

Remove all dense and sparse eligibility checks equivalent to
`capability == (8, 9)`. Compute capability may still select a genuine external kernel
family, but it must not decide whether a producer contract matches.

Literal `sm89` names may remain only when they are real third-party module, extension,
or symbol names needed to call that ABI. They must not appear in user-facing
eligibility rules, fallback reasons, or fused-projector policy.

### Upstream V lifetime/layout

Do not version-gate on ComfyUI PR #15705; it closed unmerged. Probe the installed H3
forward source for the exact known view-retention form:

- known old form: install the narrow V clone/layout shim;
- known fixed form: do nothing;
- unknown or unavailable source: fail closed to upstream behavior and report that the
  shim was not applied.

The chunked Kitchen path owns its own V lifetime and does not use this shim.

## User-facing behavior

- Preserve node IDs, inputs, order, values, defaults, and saved-workflow compatibility.
- Keep `H3 Memory Optimization.fused_qkv` values `auto` and `off`.
- Dense `auto` uses chunked Kitchen QKV when the complete producer contract matches.
- Sparse `auto` uses native fused QKV only when the complete sparse contract matches.
- `off` uses standard QKV for both dense and sparse paths.
- Applying the Memory Optimization and Sparse Attention nodes in either order must
  produce the same result.
- An explicit official Model Attention Backend override wins whether it is applied
  before or after the H3 node.
- The token refiner stays on upstream Comfy attention and standard QKV.
- Status reports the chosen provider, 4K chunk policy and Kitchen ABI, or the precise
  standard-fallback reason.
- Tooltip text: `auto uses chunked Comfy Kitchen QKV for compatible dense H3 and
  native-carrier fused QKV for compatible Sparse Sage; off uses standard QKV.`

## Execution plan

### Phase 1: Preserve a fair reference

1. Keep the legacy dense fused-QKV and dense Sage path executable only for the
   comparison harness.
2. Keep the persistent exact-shape benchmark and its CPU contract tests.
3. Label preparation-only, carrier, attention, full-block, and workflow measurements
   separately.
4. Use one enclosing CUDA event and one peak reset for each complete measured path; do
   not add separately timed asynchronous stages.
5. Do not expand the legacy architecture matrix or install an experimental Kitchen
   build into live Comfy2.

Exit: all three benchmark labels reach their intended implementation and the legacy
reference cannot become an accidental production fallback.

### Phase 2: Establish the public Kitchen producer API

Work in an isolated Kitchen checkout pinned to the intended release baseline.

1. Expose a versioned producer specification containing every derived carrier field.
2. Expose Kitchen-owned absolute K-anchor positions and anchor selection.
3. Expose Q/K chunk packing into Kitchen-owned destinations with offset, alignment,
   bound, and final-partial validation.
4. Expose V-only global quantization using Kitchen's existing two-pass semantics.
5. Expose finalization that validates every carrier field consumed by attention.
6. Test exact full-versus-external carrier and attention-output parity for:
   - short unmasked CTA-K 64;
   - long unmasked CTA-K 128;
   - masked CTA-K 64;
   - a final partial chunk.
7. Submit the API upstream or otherwise establish it as a supported Kitchen release
   contract before production cutover.

Exit: Kitchen, not H3-Optimizations, defines and validates every dense carrier field.

### Phase 3: Integrate chunked H3 QKV

1. Probe the complete public producer contract at application time.
2. Project Kitchen-provided anchor rows and apply K RMSNorm/RoPE at their absolute
   positions before Kitchen selects the anchor.
3. Allocate the final Kitchen Q/K carriers and one full BF16 V buffer.
4. Process the sequence in 4,096-row chunks:
   - Kitchen ConvRot QKV projection;
   - Kitchen RMSNorm plus position-correct RoPE on Q/K;
   - Q/K writes into their final carrier slices;
   - V copy into the retained buffer;
   - immediate release of floating Q/K/V chunk views.
5. Ask Kitchen to quantize V, release BF16 V immediately, and finalize the carrier.
6. Call Kitchen prequantized INT8 attention and preserve upstream output shape/dtype.
7. Patch main H3 blocks only and preserve the captured upstream forward as fallback.
8. Decline for training, patched/unsupported QKV layouts, missing APIs, ABI mismatch,
   non-CUDA producer support, or any incomplete contract.

Exit: compatible dense H3 reaches chunked Kitchen QKV; all unsupported cases reach the
ordinary upstream path without touching legacy dense Sage.

### Phase 4: Replace SM89 gates with contracts

1. Remove dense compute-capability eligibility checks.
2. Extend `SparseSageKernelSpec` to describe the complete carrier consumer contract.
3. Make the sparse fused producer match every field and required callable.
4. Add table-driven acceptance for a matching non-SM89 contract and fallback tests for
   every geometry, scale, V, summary, accumulator, or callable mismatch.
5. Classify every remaining `SM89`, `sm89`, `(8, 9)`, and `8.9` occurrence as either:
   - a real external ABI name used by sparse architecture dispatch; or
   - obsolete policy/reference code that must be removed.

Exit: repository search finds no SM89-only policy or error text, and tests prove
matching non-SM89 acceptance plus safe mismatch fallback.

### Phase 5: CPU and static acceptance

Run with CUDA hidden and verify:

- chunk boundaries, final partials, tile alignment, and absolute RoPE slices;
- producer ABI and every specification mismatch;
- fake-Kitchen anchor and destination offset plumbing;
- retained-V lifetime and release order;
- explicit attention-override precedence in both node orders;
- dense `auto`, `off`, and every standard-fallback reason;
- W4A8 and other unsupported QKV layouts remain standard;
- sparse matching and mismatch tables, including non-SM89 acceptance;
- node-schema and tooltip/status compatibility;
- V-layout probe behavior for known old, known fixed, and unknown upstream source;
- source isolation and absence of private Kitchen calls;
- compilation, focused tests, full pack tests, and `git diff --check`.

Exit: CPU/static tests pass without initializing CUDA and production imports do not
depend on the isolated Kitchen checkout or legacy dense modules.

### Phase 6: Authorized GPU cutover benchmark

Every GPU run requires renewed user authorization and an idle `nvidia-smi` preflight.

On the exact 54,006-token H3 block-0 shape, compare:

1. full upstream QKV -> full Kitchen prequantization -> Kitchen INT8 attention;
2. 4K chunked Kitchen QKV -> Kitchen INT8 attention;
3. legacy fused QKV -> dense Sage attention.

For each path:

- compile/warm up outside measurements;
- prove kernel reachability for the label;
- compare all carrier tensors where contracts overlap;
- report explicit numerical tolerances for non-bit-exact values;
- compare complete attention output;
- measure the entire QKV-to-attention boundary with one CUDA event and peak reset;
- report median and dispersion from steady-state samples.

Then run one real H3 block and a fixed workflow with checkpoint, seed, steps,
resolution, and attention settings held constant. Compare outputs numerically and
visually, paying particular attention to Kitchen INT8 V versus legacy Sage FP8 V.

Cutover requires all of the following:

- peak allocation delta no greater than 2.818 GiB on the original case, retaining at
  least 80 percent of the measured 1.442 GiB legacy saving;
- end-to-end time within 5 percent of legacy fused Sage, or faster;
- Kitchen-reference quality within agreed numerical and visual tolerances;
- only supported public Kitchen producer/consumer calls;
- safe standard fallback for every incomplete capability contract.

Repeat runtime validation on each architecture/backend claimed as validated. A matching
but untested specification may be described as contract-supported, not
runtime-validated.

Exit: complete-boundary and workflow evidence passes every threshold.

### Phase 7: Cut over and delete the legacy dense stack

Only after Phases 2 and 6 pass against a supported Kitchen API:

1. Keep chunked Kitchen QKV as compatible dense `auto`.
2. Keep upstream H3 QKV plus Comfy-selected attention as the only dense fallback.
3. Delete package-owned dense Sage architecture discovery, V preparation, kernel
   adapters, compatibility patches, and installation guidance.
4. Delete the legacy dense fused-QKV projector, Triton kernel, carrier contract,
   provider labels, status, and tests that exist only for it.
5. Remove token-refiner Sage pinning and any attention-function identity inspection.
6. Retain shared helpers only when a sparse path still imports them.
7. Keep the exact-shape benchmark as a Kitchen regression and tuning harness.
8. Re-run the complete CPU/static and authorized GPU matrix after deletion.

Likely deletion inventory, subject to import re-evaluation:

- `attention/sage_mem_eff.py`;
- `attention/sage_arch/`;
- dense-only `attention/sm89_compat.py` and `attention/v_snapshot_compat.py`;
- `dense_backend.py`;
- `dense_fused_qkv.py`;
- `dense_fused_qkv_kernel.py`;
- `dense_fused_qkv_contract.py`;
- dense-only provider exports, tests, status, and README text;
- `triton_i64.py` only if no retained sparse path uses it.

Exit: no package-owned dense attention or dense fused-QKV production path remains.

## Final acceptance checklist

- Compatible dense H3 uses 4K chunked Kitchen QKV and Kitchen dense INT8 attention
  without materializing full BF16 Q or K.
- Only one full BF16 V buffer survives the chunk loop and is released immediately after
  Kitchen V packing.
- Chunked carrier and output parity match the full Kitchen reference.
- Kitchen owns anchor semantics, carrier geometry, validation, V packing, and dense
  architecture dispatch.
- H3 production code contains no private Kitchen `_C` call.
- Explicit Comfy attention overrides win before and after the H3 node.
- Unsupported formats and contracts use upstream QKV.
- Token refiner behavior remains upstream-owned.
- Sparse standard and optional native fused QKV paths remain functional.
- No SM89-only dense or sparse eligibility gate remains.
- Node schema and saved workflows remain compatible.
- Complete memory, timing, kernel-reachability, numerical, and visual thresholds pass.
- Legacy dense Sage and fused-QKV code is deleted after, and only after, those gates.

## Rollback and non-goals

Rollback is always upstream H3 QKV plus normal Comfy/Kitchen attention.

Do not:

- write new dense-attention kernels;
- maintain a package-owned five-architecture dense QKV matrix while the Kitchen route
  satisfies the acceptance gates;
- stream V with per-chunk scales, because Kitchen V uses global scaling;
- run a second full QKV projection to finalize metadata;
- retain full BF16 Q or K;
- expose a chunk-size socket without evidence that users need it;
- make Sparse Sage depend on dense Kitchen carrier support;
- install the experimental Kitchen checkout into live Comfy2;
- treat PR #15705, a version string, a symbol name, a successful Triton compile, or a
  compute capability alone as proof of compatibility.

If Kitchen cannot publish/support the producer API or the complete implementation misses
the cutover thresholds, stop. Write a separate carrier-spec-driven plan for
package-owned QKV producers; do not silently depend on Kitchen private ABI or grow an
unbounded architecture matrix inside this plan.
