Yes. The clean replacement is feasible, and it can be done without rewriting the quality-sensitive routing logic. The important design is to replace **Sparse Sage as the execution backend**, not “replace sparse attention” wholesale.

The final target should be:

```text
                         H3 Sparse Attention
                                  |
                    existing H3-specific router
                                  |
                 +----------------+----------------+
                 |                                 |
          standard QKV                      fused ConvRot QKV
                 |                                 |
        floating Q/K/V                  INT8 Q/K + scales + V
                 |                         + router summaries
                 |                                 |
         portable Triton                    portable Triton
        floating sparse kernel             INT8 sparse kernel
                 |                                 |
                 +----------------+----------------+
                                  |
                              H3 output
```

`spas_sage_attn` disappears entirely. Triton becomes the only additional sparse execution dependency.

The existing router remains the source of sparse semantics: only pure-video→video blocks are reduced, all non-video KV remains visible, non-pure-video queries remain dense, and the retained video budget remains fixed-density.

## 1. Define exactly what is being replaced

Today the Sparse Sage layer does four distinct jobs:

1. Resolves the correct `spas_sage_attn` binary/kernel for SM80/86/87/89/90/120.
2. Converts standard floating Q/K/V into Sage-specific Q/K and V carriers.
3. Validates those carriers against the architecture-specific ABI.
4. Executes the compiled Sparse Sage kernel.

The H3 router is separate and should stay separate.

The native sparse fused-QKV producer is also already separate: it emits INT8 Q/K, block scales, floating V, and Q/K routing summaries.

Therefore the migration should remove:

```text
SparseSageKernelSpec
spas_sage_attn module discovery
split/monolithic extension handling
Sage kernel symbol lookup
Sage FP8-V preparation
Sparse Sage executor
Sparse Sage custom op
architecture-specific Sage sparse ABI selection
```

while preserving:

```text
SparseTileRouter
packed H3 runtime layout
video-budget policy
early/late density policy
fused sparse QKV
fused routing summaries
standard QKV fallback
```

That boundary is important.

---

# Phase 0 — Freeze the current behavioral baseline

Before replacing anything, create a benchmark/reference harness around the current implementation.

Record current Sparse Sage behavior for at least:

```text
SM89 / RTX 4070 or 4090
Q tile = 128
KV tile = 64
BF16 H3
sequences:
    small synthetic
    tile-boundary cases
    ~32k
    ~54k real H3
densities:
    10%
    25%
    50%
    100%
```

Capture separately:

* router output;
* selected video KV indices;
* `valid_block_num`;
* sparse attention output;
* sparse-kernel time;
* complete attention time;
* fused-QKV + sparse time;
* block time;
* local allocation peak;
* complete workflow peak;
* output numerical difference from dense attention.

Do not use visual generations as the first correctness test. The router and kernel should be independently testable.

The benchmark harness should call internal functions directly. Do **not** add a public node option selecting “Sparse Sage” versus “Triton.”

Keep the old Sparse Sage implementation during development solely as a reference implementation. Delete it after the Triton backend passes the final gates.

---

# Phase 1 — Introduce a package-owned sparse execution contract

The new sparse backend should no longer inherit its geometry and memory layout from Sparge.

Create something approximately like:

```python
@dataclass(frozen=True)
class SparseTritonSpec:
    q_tile: int
    kv_tile: int
    head_dim: int

    float_layout: str
    fused_qk_layout: str
    output_layout: str

    supports_float: bool
    supports_int8_qk: bool

    launch_configs: tuple
```

Initially:

```text
q_tile    = 128
kv_tile   = 64
head_dim  = 128
```

This is already the validated H3 routing geometry on SM89 and matches the current fused sparse QKV producer.

Critically, these become **our sparse algorithm's geometry**, not “the dimensions the installed Sparge binary happens to require.”

This has an important consequence: architecture support becomes much simpler. A 4090 and a 5090 can use the same logical 128Q×64KV sparse algorithm even if Triton chooses different warps/stages when compiling it.

The spec should describe semantic layout, not NVIDIA architecture:

```text
float path:
    Q/K/V floating BLHD
    128 × 64 sparse tiles

fused path:
    Q/K HND INT8
    one Q scale per 128-token block
    one K scale per 64-token block
    V HND BF16/FP16
    router summaries already available
```

The backend preflight decides whether the active Triton/device runtime can execute those contracts.

---

# Phase 2 — Preserve the current router unchanged initially

This is where I would avoid unnecessary architectural cleanup.

The current router emits:

```text
lut
valid_block_num
metadata
```

Its LUT indices are delta-encoded for Sparse Sage, but that **does not require changing the router**.

The Triton loop can simply do:

```python
key_block = 0

for slot in selected_blocks:
    key_block += load(lut[slot])
    ...
```

instead of loading an absolute key-block index.

The existing encoding is actually useful:

```text
dense:
    [0, 1, 1, 1, 1, ...]

selected:
    prefix deltas
    + sorted video-block deltas
```

The kernel can consume it directly.

That means there is no need to touch:

* pooling;
* video boundary calculation;
* top-k;
* retained tile rounding;
* pure-video detection;
* fixed-density semantics;
* summary-based routing.

The current router already calculates exactly the information needed to split dense prefix queries from sparse pure-video queries.

This substantially lowers migration risk.

Later, after Sparse Sage has been deleted, we can decide whether delta encoding still makes sense. That should not be part of the initial replacement.

---

# Phase 3 — Implement the floating-QKV Triton sparse kernel

This should be the first new execution kernel because it is the easiest path to validate independently.

PlagueKind provides a very useful starting reference. Their current kernel is a vendored/reduced LightX2V Triton block-sparse forward implementation, operates on floating Q/K/V, performs online softmax in FP32, and has already needed only launch-configuration changes to work on consumer Blackwell.

If code is adapted from it, retain the appropriate Apache-2.0 attribution/license notices.

### Input layout

For standard H3 QKV, use **BLHD**:

```text
[B, sequence, heads, head_dim]
```

rather than forcing HND contiguous storage.

This is exactly the optimization PlagueKind identified: H3 originally creates `[S,H,D]`; the HND tensor seen by attention is a transpose view. Transposing it back to BLHD restores the original contiguous layout without copying. Their comments explicitly note that materializing HND would otherwise be very expensive at long H3 sequence lengths.

Thus standard sparse execution becomes:

```python
q_blhd = q_hnd.transpose(1, 2)
k_blhd = k_hnd.transpose(1, 2)
v_blhd = v_hnd.transpose(1, 2)
```

and normally all three are zero-copy views onto the original QKV storage.

Require them to be contiguous; if upstream changes layout and they are not, decline the sparse path rather than blindly copying several gigabytes.

### Mathematical kernel

For every query tile:

```text
load Q tile

for each selected KV block:
    load K tile
    logits = Q @ K^T * scale

    update online FP32 softmax:
        running max
        running denominator
        rescale old accumulator

    load V tile
    accumulator += softmax_probability @ V

normalize
store output
```

Use:

```text
Q/K/V input      BF16 or FP16
QK accumulation  FP32
softmax state    FP32
output accum     FP32
stored output    input dtype
head dim         128
```

The numerical reference should be ordinary PyTorch attention restricted to exactly the same selected key tokens.

### Tail handling

Every masked Q/K/V load must specify `other=0.0`.

PlagueKind explicitly fixed this because Triton masked lanes are otherwise undefined and can poison reductions with NaNs.

Test all:

```text
S % 128 = 0
S % 128 = 1
S % 128 = 127

S % 64 = 0
S % 64 = 1
S % 64 = 63
```

### Dense prefix versus sparse video

Do **not** make every program iterate up to the maximum LUT width.

Our H3 routing has two regimes:

```text
prefix/mixed query blocks:
    attend every KV block

pure-video query blocks:
    attend every non-video KV block
    + retained video KV blocks
```

There are two implementation candidates.

**Preferred implementation: two launches.**

Launch A:

```text
query blocks [0, pure_video_q_start)
dense KV traversal
```

It need not load the LUT at all; it just visits sequential KV blocks.

Launch B:

```text
query blocks [pure_video_q_start, end)
delta-LUT traversal
```

All pure-video query rows have the same number of selected blocks under fixed-density routing, so they share one sparse loop bound.

This avoids paying a full-KV loop on every sparse video query.

A one-launch variable-`valid_block_num` implementation is worth retaining as a correctness/reference prototype, but the two-launch implementation is more likely to be performant.

---

# Phase 4 — Add launch portability without architecture-specific source kernels

Use one Triton source kernel and a **configuration ladder**, not SM-specific Python implementations.

PlagueKind already demonstrates this pattern. Their 128×64 kernel tries different warp/stage configurations because LightX2V's original settings exceed the 5090's available shared memory.

Our kernel should have something like:

```python
FLOAT_128x64_CONFIGS = (
    (8, 3),
    (4, 3),
    (8, 2),
    (4, 2),
    (4, 1),
)
```

The precise list must come from profiling, not this example.

Key the cache by something like:

```text
device capability
dtype
head_dim
Q tile
KV tile
```

For the first launch:

1. try preferred configuration;
2. if Triton reports deterministic `OutOfResources`, try the next;
3. cache the first viable configuration;
4. if no candidate launches, mark Triton sparse unsupported for that execution environment and fall back dense.

Once a configuration has successfully launched, later unexpected errors are real kernel failures, not compatibility fallback.

This distinction keeps failures debuggable.

Do not initially use a giant `triton.autotune` search. Compilation overhead on a huge H3 workload would be obnoxious. A small curated ladder is sufficient for portability.

Performance tuning can come later.

---

# Phase 5 — Replace `SparseSageExecutor` with `SparseTritonExecutor`

The existing high-level backend structure can remain.

Today:

```text
HybridSparseBackend
    ↓
SparseTileRouter
    ↓
SparseSageExecutor
```

Change it to:

```text
HybridSparseBackend
    ↓
SparseTileRouter
    ↓
SparseTritonExecutor
```

The `HybridSparseBackend` already cleanly separates routing from consumption.

The new executor needs two preparation entry points:

```python
prepare(q, k, v, ...)
prepare_projected(projected, ...)
```

### `prepare()` — standard QKV

Store only:

```text
Q/K/V floating views
LUT
routing metadata
dtype/layout
```

Do **not** quantize Q/K.

Do **not** construct FP8 V.

Do **not** allocate Sage carriers.

This may actually reduce memory compared with the current non-fused Sparse Sage path because the current executor first creates V carriers and Q/K INT8 carriers before execution.

### `execute()` — standard QKV

Transpose HND views back to BLHD and call the floating Triton kernel.

Return either BLHD → HND view or output HND according to the existing backend contract.

No intermediate contiguous HND Q/K/V copies.

### Metadata

Replace status fields such as:

```text
sparge_attention
sparse_v_format
sparse_extension_layout
sparse_v_quant_bound
```

with:

```text
backend = triton
kernel = float_sparse
q_tile
kv_tile
launch_warps
launch_stages
dtype
triton_version
```

Keep density/routing metadata unchanged.

---

# Phase 6 — Preserve fused sparse QKV with a second INT8 Triton kernel

This is the important phase for retaining our memory optimizer.

We should **not** make the fused sparse QKV path round-trip through floating Q/K.

The existing fused projector already gives us almost perfect inputs for a package-owned sparse Triton kernel:

```text
q_int8
k_int8
q_scale
k_scale
V
q_summary
k_summary
```

with:

```text
Q block = 128
K block = 64
```

and one scale per head/block.

The new INT8 sparse attention kernel can consume those carriers directly.

### QK math

For one selected Q/K block:

```python
score_int32 = dot(q_int8, k_int8)

score_fp32 = (
    score_int32
    * q_scale[q_block]
    * k_scale[k_block]
    * head_dim**-0.5
)
```

Because the current fused carrier uses one scalar per Q tile and one scalar per K tile, this is particularly simple.

Then run the same online FP32 softmax/PV accumulation used by the floating kernel.

V remains BF16/FP16.

No:

```text
Q dequant tensor
K dequant tensor
FP8 V carrier
Sage _fused extension
architecture-specific Sparse Sage binary
```

is required.

### Reuse the existing summaries

`prepare_projected()` continues to call:

```python
router.build_lut_from_summaries(
    projected.q_summary,
    projected.k_summary,
    ...
)
```

exactly as it does today.

Therefore fused QKV continues to avoid rebuilding floating Q/K merely for routing.

### Layout

Do not force the fused carrier through BLHD merely for symmetry.

The existing fused producer already emits contiguous HND carriers. The INT8 Triton kernel should initially consume:

```text
Q INT8 HND
K INT8 HND
V HND
```

directly.

Thus we probably end up with:

```text
one floating BLHD Triton sparse kernel
one INT8-HND Triton sparse kernel
```

They can share the same online-softmax design and test reference, but different memory layouts are justified because they prevent huge copies.

This is still dramatically simpler than per-architecture Sparse Sage binaries.

---

# Phase 7 — Generalize the existing fused sparse QKV beyond SM89

Today fused sparse QKV is restricted to SM89 because the **consumer** was an SM89 Sparse Sage ABI, not because the H3 projection mathematics are inherently Ada-specific.

Once the consumer is our own Triton INT8 kernel, define a package-owned fused sparse contract:

```text
head_dim = 128
Q block = 128
K block = 64
Q/K signed INT8
one FP32 block scale
V = BF16/FP16 HND
summaries = FP32/BF16 as currently validated
```

Then the current fused projector can be tested on additional GPUs without translating its output into different Sage ABIs.

The eligibility test becomes:

```text
ConvRot-256 TensorWise INT8 weights
+ Triton available
+ device supports required INT8 tl.dot
+ fused QKV kernel launch validated
+ INT8 sparse kernel launch validated
```

rather than:

```text
capability == SM89
+ matching Sparge kernel
+ matching Sparge carrier ABI
```

This could be one of the biggest payoffs of the migration.

Validate independently on:

```text
SM80
SM86
SM89
SM90
SM120/121
```

SM87 if hardware/testing access exists.

Unknown/future devices remain standard-QKV + floating sparse only if that path is validated there; otherwise dense fallback.

No guessing.

---

# Phase 8 — Decide whether 128Q×64KV should remain universal

For the first replacement, I strongly recommend keeping:

```text
128Q × 64KV
```

everywhere.

Reasons:

* it preserves the current validated SM89 routing;
* it matches the current fused QKV summaries/carriers;
* PlagueKind reports 128×64 as its fastest measured configuration on the 5090;
* it dramatically simplifies the first portability implementation.

Current Sparse Sage changes to 64Q×128KV on SM90 because that is what the Sage SM90 binary expects.  Once Sparse Sage is gone, that is no longer automatically our problem.

Benchmark 128×64 on H100/H200.

Only if SM90 performance is materially poor should we parameterize the package-owned Triton backend for:

```text
64Q × 128KV
```

That can still use the **same Triton source** with constexpr tile dimensions.

The fused projector would then need a second summary/scale geometry, but that should be added because measurements justify it, not because Sage used it.

---

# Phase 9 — Capability resolution and fallback

Sparse Attention should become optional by capability, while workflows remain portable.

Resolution should be:

```text
resolve normal dense candidate first

if sparse not requested:
    use dense candidate

if sparse requested:
    if Triton sparse float contract unsupported:
        use dense candidate
        report reason

    otherwise:
        select Triton sparse

        if fused QKV requested:
            if fused producer + INT8 sparse consumer supported:
                select fused sparse
            else:
                use standard QKV + floating Triton sparse
```

That gives graceful degradation:

```text
best:
    fused QKV + INT8 Triton sparse

next:
    standard QKV + floating Triton sparse

fallback:
    normal dense attention
```

A failure to fuse QKV must **never disable sparse attention**.

A failure to sparsify must **never prevent stock H3 execution**.

No additional UI settings are needed.

---

# Phase 10 — Remove token-refiner coupling

This migration is a good point to finish the earlier cleanup.

The new sparse kernel should only patch the main H3 DiT blocks.

The token refiner stays on whichever dense backend Comfy has selected.

Remove:

```python
requires_registered_sage = True
```

and all token-refiner Sage pinning.

The Triton sparse implementation has no reason to affect unrelated attention calls.

---

# Phase 11 — Correctness test hierarchy

### A. Router tests — CPU

Do not change expected outputs from current routing.

Test:

* exact packed-layout boundaries;
* target video must remain the final segment;
* pure-video query start;
* pure-video KV start;
* fixed budget rounding upward;
* 1% minimum;
* 100% density;
* partial tiles;
* audio/text/ref always visible;
* summary path and floating path produce the same selection;
* fused summaries reproduce the expected route.

### B. Floating Triton kernel — synthetic GPU

Generate random BF16/FP16 tensors.

For each LUT row, construct a PyTorch reference:

```python
selected_k = concatenate(selected K blocks)
selected_v = concatenate(selected V blocks)

reference = softmax(
    q @ selected_k.T * scale,
    dim=-1,
) @ selected_v
```

Compare against Triton.

Test:

```text
B=1
H=1, 8, 56
D=128

Q blocks:
1
2
many

K blocks:
1
2
many

tails:
all combinations around 64/128 boundaries
```

Test dense LUT as well. If every block is selected, sparse output should match normal dense attention within the expected BF16/FP16 numerical tolerance.

### C. INT8-carrier Triton kernel

The reference should reconstruct the exact fused-carrier math:

```python
q_approx = q_int8 * q_scale
k_approx = k_int8 * k_scale
```

and then apply ordinary sparse attention over the selected blocks.

This isolates the attention kernel from projection quantization error.

Triton must match that reference closely.

### D. Fused QKV + Triton

Compare:

```text
current fused QKV → Sparse Sage
versus
same fused QKV → INT8 Triton
```

using identical router output.

Any difference now comes from the sparse attention consumer, not the projector or router.

### E. Route equivalence

Record current Sparse Sage LUT selections before migration.

The Triton backend must consume **exactly those selections**.

Do not allow a faster kernel implementation to quietly change routing.

### F. Full H3

Then test:

```text
T2V
FL2VA
Ref2VA if appropriate
long sequences
audio
mixed packed segments
```

with fixed seeds.

---

# Phase 12 — Performance and memory acceptance

Measure five paths where available:

```text
1. dense reference
2. current Sparse Sage + standard QKV
3. current Sparse Sage + fused QKV
4. Triton sparse + standard QKV
5. Triton sparse + fused QKV
```

At identical video density.

Measure separately:

```text
router
QKV
carrier preparation
sparse attention kernel
complete attention
complete DiT block
denoise step
workflow
```

For memory:

```text
local attention peak
full block peak
full denoise peak
full workflow peak
```

For the fused path, the acceptance criterion should explicitly include preserving the current fused-QKV memory benefit. A kernel that runs but causes full floating Q/K to reappear has failed the design goal.

The most important comparison is:

```text
old fused QKV + Sparse Sage
vs
fused QKV + INT8 Triton sparse
```

Ideally the Triton version will actually save additional memory because the current Sparse Sage path still converts V into an architecture-specific FP8/FP16 carrier. The new kernel can consume the fused projector's existing V directly.

---

# Phase 13 — Performance tuning after correctness

Only after outputs and memory are correct:

### Tune warp/stage ladders

Benchmark representative devices:

```text
SM80
SM86
SM89
SM90
SM120
```

Keep tuning data out of the algorithm.

Something like:

```python
CONFIGS_128x64 = (
    TritonLaunch(warps=8, stages=3),
    TritonLaunch(warps=4, stages=3),
    ...
)
```

is acceptable.

Do not create:

```python
if sm89:
    execute_sm89_kernel()
elif sm90:
    execute_sm90_kernel()
```

unless the actual algorithm must differ.

### Separate floating and INT8 tuning

Their best configurations may differ substantially.

Cache independently:

```text
float-128x64
int8-128x64
```

per device.

### Avoid compilation explosions

Do not make sequence length, exact retained block count, or video budget constexpr unless profiling proves it valuable.

Those values vary constantly across H3 workflows.

Compile on stable structural values:

```text
D
Q_TILE
KV_TILE
dtype/mode
launch config
```

and keep sequence/top-k runtime values where practical.

---

# Phase 14 — Remove `spas_sage_attn`

Only after the Triton path passes the required GPUs.

Delete from `sparse_sage.py` or delete the file entirely:

```text
importlib.metadata lookup for spas-sage-attn
_load_qattn_surface
_load_fused_surface
SparseSageKernelSpec
SparseSageExecutor
prepare_sparse_sage_v
sparse_sage_attention_op
all Sage kernel names
SM-specific Sparge maps
extension_layout
v_quant_bound
Sparge status fields
```

The current file exists largely to negotiate the compiled Sparge ABI.

Remove:

* `spas_sage_attn` installation documentation;
* troubleshooting instructions for compiled Sparse Sage;
* split/monolithic wheel compatibility logic;
* tests for extension symbol discovery;
* tests for Sage architecture lookup.

Replace them with:

```text
Triton available?
supported execution device?
known viable launch configuration?
float sparse supported?
INT8 sparse supported?
```

That is a dramatically smaller compatibility surface.

---

# Recommended file structure

I would end up with something approximately like:

```text
h3_optimizations/
  attention/
    sparse/
      backend.py
      config.py
      router.py

      triton_spec.py
      triton_float.py
      triton_int8.py
      triton_executor.py
      triton_launch.py

      fused_qkv.py

      stats.py
```

`backend.py` remains the high-level H3 integration.

`router.py` remains the sparse policy.

`triton_float.py` owns floating block-sparse attention.

`triton_int8.py` owns direct fused-carrier attention.

`triton_launch.py` owns viability/tuning/cache.

`triton_executor.py` owns tensor validation and execution orchestration.

No file should know about `spas_sage_attn`.

---

# Rollout order

I would implement it in this order:

**Milestone 1 — Floating reference backend**

```text
current router
→ new floating Triton kernel
→ standard QKV only
```

No fused integration yet. Get routing parity and numerical correctness.

**Milestone 2 — Performance-portable floating backend**

Add config ladder and validate SM89 + SM120 first because those are available/currently relevant.

Benchmark against PlagueKind and Sparse Sage.

**Milestone 3 — Native fused-carrier kernel**

```text
existing fused QKV
→ existing summaries/router
→ new INT8 Triton sparse kernel
```

This is the important production path.

**Milestone 4 — Remove SM89 restriction from sparse fused QKV**

Promote each GPU only after fused projection + INT8 sparse kernel has actually run there.

**Milestone 5 — Validate Ampere/Hopper**

SM80, SM86, SM90.

Decide from measurements whether 128×64 remains universal.

**Milestone 6 — Delete Sparse Sage**

Only then remove the dependency and ABI code.

---

# Final production behavior

The desired result is:

```text
H3 Sparse Attention
        |
        +-- Triton unavailable/unsupported
        |       → preserve dense execution
        |
        +-- Triton available
                |
                +-- standard QKV
                |       → floating Triton sparse
                |
                +-- compatible ConvRot fused QKV
                        → INT8-carrier Triton sparse
```

No Sparge installation.

No Sage sparse wheel compatibility.

No native sparse extension discovery.

No architecture-specific sparse binary ABI.

No change to saved workflows.

And most importantly, **the fused sparse path gets simpler rather than weaker**: the current Q/K block carriers and routing summaries can feed a Triton kernel directly, while V can remain in the representation the fused projector already produced instead of being converted to whatever Sparse Sage expects.

That is the version I would aim to ship.
