# QKV/MLP allocator reuse experiment

## Result: rejected, 2026-08-22

Stage A ran all five arms twice on the RTX 4070, sequence 54006, block 0,
ref2va ConvRot INT8 checkpoint. Repeats agree to the megabyte. **No candidate
beat the control on any metric, so the QKV default stays at 4096 and Stage B was
not run.** Raw JSON: `.agent\tmp\qkv_allocator\`.

| arm | predicted | reuse | reserved | inactive split | large segments | steady ms |
| --- | --- | --- | --- | --- | --- | --- |
| 4096:4096 (control) | fit with split | no | **5.883 G** | 14.7 M | **23** | 676.1 |
| 2816:4096 | fit with split | no | 5.906 G | 14.7 M | 26 | 669.4 |
| 2688:4096 | exact fit | no | 5.918 G | 14.7 M | 26 | 671.7 |
| 3072:4608 | exact fit | no | 5.916 G | 14.7 M | 26 | 672.1 |
| 2048:3072 | exact fit | **yes, 12/12** | 5.908 G | 14.7 M | 27 | 671.7 |

The decisive row is the last one. `2048:3072` achieved exactly the address
identity the hypothesis predicted — the first full-width MLP tile expansion
started at an address a QKV chunk had released, in every one of twelve
consecutive forwards — and it still reserved **25 MiB more** than the control.
Reuse happened and bought nothing. That falsifies the mechanism, not just the
particular chunk sizes.

Peak *allocated* bytes were byte-identical across every arm (5649.1 MiB),
confirming that QKV chunk rows do not touch peak allocated at all.
`inactive_split_bytes` was identical (14.7 MiB) everywhere, and
`num_alloc_retries` was 0 everywhere. Runtime spread was ~2 % with the control
not the slowest, so nothing was bought on time either.

### Why shrinking the QKV buffer makes reserved memory worse

Large-segment histogram, control versus 2688:

| segment size | control | 2688:4096 |
| --- | --- | --- |
| 168.0 MiB | 1 | — |
| 112.0 MiB | 1 | 2 |
| 38.0 MiB | 2 | 4 |
| 24.0 MiB | — | 1 |
| 22.0 MiB | 1 | — |
| 14.0 MiB | — | 1 |

A 112 MiB segment already exists in both arms and is **fully occupied by a
long-lived 110 MiB allocation** (`blocks=[(110, active), (1, active)]`). So
dropping QKV from a 168 MiB buffer to a 112 MiB one does not let it move into
that segment — it creates a *second* 112 MiB segment. The 56 MiB saved in that
size class is more than cancelled by the extra small segments that the finer
chunking's tail buffers create: 21 QKV chunks with a 246-row tail instead of 14
chunks with a 758-row tail introduces new size classes rather than eliminating
any.

The allocator model below was right about sizes and wrong about behaviour. It
correctly predicted that 2688 rows produce a segment byte-identical to the
4096-row MLP request, and that segment did appear — but an intervening
allocation claims the freed block before the MLP asks for it, so size matching
alone never determined who got the block.

Everything below is the original design, kept for provenance and because the
harness remains the right tool if this is ever revisited.

---


Does the CUDA caching allocator hand the bounded MLP the same block that a
chunked QKV projection just released, and does aligning the two chunk sizes make
that reuse happen more cleanly?

Stage A answers "does reuse occur" on one real block. Stage B answers "does it
survive the full model and change reserved VRAM". Do not change the production
QKV default before Stage B.

## Verified constants

Read from `minimax_h3_ref2va_pruned_int8_convrot.safetensors`, block 0:

| Tensor | Shape | Per-row BF16 bytes |
| --- | --- | --- |
| `attn.qkv_proj` output | `[rows, 21504]` | 43008 |
| ConvRot MLP tile expansion | `[rows, 14336]` | 28672 |

`hidden = 5376`, `fc1 = [28672, 5376]`, `fc2 = [5376, 14336]`. Each of the two
ConvRot tiles carries gate and up features for half the SwiGLU width, so one
tile expansion is 14336 wide, not 28672. The QKV/MLP per-row ratio is exactly
3:2.

Production settings this experiment holds fixed:

- `apply.py` builds `SparseFusedQKVProjector(kernel_spec, chunk_rows=4096)`.
- Auto on a ConvRot checkpoint resolves the MLP to `mlp_chunked_convrot_2slice`
  with `MemoryRequest.chunk_rows = 4096`.
- The Comfy2 server logs `Device: cuda:0 NVIDIA GeForce RTX 4070 : native`, so
  production runs the **native** caching allocator. This checkout's local
  `cuda_malloc.py` change sets
  `PYTORCH_CUDA_ALLOC_CONF=backend:native,garbage_collection_threshold:0.95,expandable_segments:False`
  under `--disable-cuda-malloc`. `bench_h3_block_allocator.py` reproduces that
  string and aborts if the active backend is not `native`.

## Four corrections to the original plan

**1. The exact-fit candidate is 2688 QKV rows, not 2816.** The native allocator
rounds a large segment up to a 2 MiB multiple. 2688 rows request 115,605,504 B
and get a 117,440,512 B segment, which is byte-identical to the 4096-row MLP
request. 2816 rows get a 121,634,816 B segment and leave a 4 MiB split
remainder. Predicted behaviour, from the allocator model in the script:

| QKV rows | MLP rows | QKV segment | MLP request | Verdict |
| --- | --- | --- | --- | --- |
| 4096 | 4096 | 168.0 MiB | 112.0 MiB | fit with 56 MiB split |
| 3072 | 4096 | 126.0 MiB | 112.0 MiB | fit with 14 MiB split |
| 2816 | 4096 | 116.0 MiB | 112.0 MiB | fit with 4 MiB split |
| **2688** | **4096** | **112.0 MiB** | **112.0 MiB** | **exact fit** |
| 2560 | 4096 | 106.0 MiB | 112.0 MiB | too small, no reuse |
| 3072 | 4608 | 126.0 MiB | 126.0 MiB | exact fit |
| 2048 | 3072 | 84.0 MiB | 84.0 MiB | exact fit |
| 4096 | 6144 | 168.0 MiB | 168.0 MiB | exact fit |

The last row matters: at the current production QKV default, an MLP running at
6144 rows is already an exact match. If reuse turns out to be the dominant
effect, raising MLP rows is an alternative to lowering QKV rows.

**2. Reuse probably already happens at 4096/4096.** A cached 168 MiB block
serves a 112 MiB request by splitting, so the current configuration likely also
reuses. The measurable difference between arms is therefore fragmentation and
segment count, not reuse versus no reuse. Treat `inactive_split_bytes` and
`segment.large_pool.current` as the quantitative result, and address identity as
the mechanism check.

**3. The MLP never allocates a `chunk_rows`-wide tensor unless a modulation
segment is long enough.** `iter_mod_chunks` refuses to cross a modulation
segment boundary, so at sequence 4096 with a 512-token text prefix and a
256-token audio prefix the MLP slabs are 512, 256, and 3328 rows and the 112 MiB
allocation never occurs. The harness defaults to `--sequence 54006` and
`--video-start 256`, giving a 53750-row video segment: thirteen full 4096-row
MLP slabs plus a tail. Do not shrink the sequence below roughly 3 × MLP rows of
video or the experiment measures nothing.

**4. Expected effect size is small and computable in advance.** All 50 blocks
reuse the same buffers in sequence, so this is a one-time reserved-VRAM
difference, not a per-block one. If QKV and MLP already share one segment the
ceiling is 168 − 112 = 56 MiB; if they do not share today it is
(168 + 112) − 112 = 168 MiB. Against a 12,282 MiB card that is 0.46 % to 1.4 %.
Decide whether that is worth Stage B before spending the GPU time.

Against that, the runtime cost is real: at sequence 54006, 2688 rows means 21
QKV chunks instead of 14, a 50 % increase in projection launches. Set the guard
threshold before running.

## Stage A: one real block

`benchmarks/bench_h3_block_allocator.py` runs one arm per process through the
real production lifecycle — AdaLN, chunked Sparse Sage QKV, real sparse
attention, residual, `del h, attn_out`, ConvRot two-slice MLP, residual — with
no shape-faithful stand-ins anywhere.

What it changes relative to `bench_h3_block.py`:

- `--qkv-chunk-rows` and `--mlp-chunk-rows` are parameters.
- Real `HybridSparseBackend` with a `required=True` projector, so a format
  downgrade raises instead of silently falling back to the dense path.
- **No `empty_cache()` inside the measured series.** One release after warmup
  gives every arm the same starting cache; nothing after that clears it.
- The working `x` is preallocated and restored with `copy_()`, so the series
  never allocates a sequence-sized tensor of its own.
- Allocator statistics, a large-segment histogram, and the block layout of the
  segments under test are the recorded results. `max_memory_allocated` and
  runtime are secondary controls.
- Address tracing wraps `qkv_proj.forward` and `memory.linear._convrot_linear`,
  recording `data_ptr()` and never retaining a tensor.

Run it behind the GPU preflight wrapper:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\.agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1 `
  -- .\custom_nodes\H3-Optimizations\benchmarks\run_qkv_allocator_sweep.py `
     --checkpoint hf_minimax_h3\minimax_h3_ref2va_pruned_int8_convrot.safetensors `
     --result-dir .\.agent\tmp\qkv_allocator `
     --repeats 2 `
     --i-understand-this-uses-gpu
```

The driver runs each arm in a fresh interpreter. Default arms:
`4096:4096` (production control), `2816:4096`, `2688:4096` (predicted exact
fit), `3072:4608` and `2048:3072` (exactly matched pairs, mechanism probes, not
production candidates). The two exact-match arms move MLP rows away from the
measured optimum on purpose: if an exactly matched pair produces no reuse
benefit, no near match will either, and the hypothesis is dead without further
testing.

### Primary observable

`reuse.first_full_fc1_reuses_qkv_address` per forward: whether the first
full-width MLP tile expansion starts at an address a QKV projection chunk used
and released earlier in the same forward. This is a binary fact about the
allocator, not an inference from a byte total.

### Secondary metrics, read after the last forward

- `reserved_bytes.all.current` and `.peak`
- `inactive_split_bytes.all.current` and `inactive_split.all.current`
- `segment.large_pool.current`
- `num_alloc_retries`, `num_ooms`
- the `segments.under_test` block layout for the predicted QKV and MLP segment
  sizes
- `timing.steady_median_ms` as the cost guard

### Decision rule

Adopt a candidate for Stage B only if, against the `4096:4096` control and
reproduced across both repeats:

- steady-state `reserved_bytes.all.current` drops by at least 32 MiB, **and**
- `inactive_split_bytes.all.current` does not increase, **and**
- `steady_median_ms` regresses by no more than 2 %.

If the arms converge to the same reserved bytes, inactive split bytes, and
segment count after a few forwards, the alignment hypothesis is dead: leave the
QKV default at 4096 and stop. Record that result — it is a real answer.

### Diagnostic pass

Run separately, never mixed with timing results, because history recording adds
overhead:

```powershell
... bench_h3_block_allocator.py --qkv-chunk-rows 2688 --mlp-chunk-rows 4096 `
    --forwards 3 --record-history --snapshot-out .\.agent\tmp\qkv2688.pickle `
    --i-understand-this-uses-gpu
```

Load the pickle in PyTorch's memory viewer and check the concrete claim: under
the winning arm, does the block created during QKV become the block serving the
first full MLP expansion, while the control leaves a larger size class split or
idle beside a separately created 112 MiB class?

## Stage B: full H3 denoiser

Only after a Stage A winner. A single `DiTBlock` does not reproduce Comfy's
whole-model weight and offload lifecycle, so it cannot establish the
user-visible reserved-VRAM benefit.

Prerequisite, a behaviour-preserving refactor of `h3_optimizations/apply.py`:
replace the literal `chunk_rows=4096` at the `SparseFusedQKVProjector`,
`FP8SparseQKVProjector`, and `TritonSparseQKVProjector` call sites with a module
constant `SPARSE_QKV_CHUNK_ROWS = 4096`, so Stage B is a one-line edit rather
than three. This is intentionally not applied yet — the Stage A baseline should
run against unmodified production code.

Procedure:

1. Idle-GPU preflight, then start Comfy2 normally.
2. Run a real Ref2VA generation at a production sequence with Sparse Attention
   and Memory auto, at the control value. Record peak reserved VRAM across the
   whole sampling run, plus wall-clock.
3. Restart the server, set `SPARSE_QKV_CHUNK_ROWS` to the Stage A winner, repeat
   the identical workflow and seed.
4. Repeat each condition at least twice, alternating, and run several
   consecutive generations per server start with no manual `/free` between them
   — the question is whether reuse survives repeated requests.

Stage B succeeds only if peak reserved VRAM drops by a margin larger than the
run-to-run spread of the control, at no more than a 2 % wall-clock cost. Change
the production default only then.

## GPU safety

Every command here allocates VRAM. Each one needs explicit permission for that
specific run and must go through
`.agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1`. Judge
idle from power draw and clocks, not `utilization.gpu`.
