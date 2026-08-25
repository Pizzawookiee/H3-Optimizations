# H3 Optimizations

Standalone production optimization nodes for MiniMax H3 in ComfyUI.

This pack owns its native Kitchen INT8 attention kernels, chunked QKV producer,
sparse routing, bounded MLP execution, and an optional DynamicVRAM safety
control.

## Nodes

- H3 Memory Optimization selects the resolved dense H3 attention path and
  applies compatible memory/execution providers. ConvRot INT8 QKV can project
  in 4K token chunks directly into Comfy Kitchen carriers. Native FP8 uses held
  chunked FP8 execution, and ordinary BF16/FP16 QKV/MLP weights may be converted
  to FP8 E4M3 when accelerated FP8 is available. FinalLayer norm, modulation,
  and FP32 output projection also run in bounded token chunks using the same
  activation chunk-row setting. The advanced `Preserve
  precision` toggle forbids new weight quantization. With sparse Kitchen,
  checkpoint-native ConvRot INT8 QKV streams BF16 projection, norm, and RoPE
  chunks into the routed INT8 carrier instead of materializing full-sequence
  BF16 Q/K/V. Other dense QKV stays on the upstream Comfy path, while MLP auto
  keeps BF16/FP16 weights floating and still uses bounded token chunking.
  Checkpoint-native ConvRot, FP8, and W4A8
  remain native where a compatible bounded MLP provider exists; unsupported
  quantized formats preserve upstream Comfy execution. Preserve precision
  defaults on for newly added nodes and omitted API inputs. Workflows that
  explicitly store `false` retain the quantizing auto policy.
- H3 AIMDO Residency Limiter helps prevent avoidable out-of-memory errors on
  long or high-resolution H3 videos. ComfyUI can underestimate how much working
  memory H3 will need and keep too much of the model in VRAM. The generation
  can then run out of memory even when that resolution and length would fit
  with less of the model kept on the GPU. This node restricts how much of the
  H3 model ComfyUI keeps in VRAM, leaving more room for the video itself and
  other temporary work. It changes memory management, not the model math or
  video quality.

  `2 blocks` is the recommended starting point. Lower settings leave more free
  VRAM but can be slower because ComfyUI must stream more model weights as they
  are needed. Higher settings keep more of the model ready on the GPU, which
  can be faster but leaves less room for the video. `0 blocks` is the most
  aggressive limit; `stock` turns the limiter off and restores ComfyUI's normal
  behavior. AIMDO is the model-weight streaming system used by DynamicVRAM; the
  numeric settings require DynamicVRAM with async weight offloading. The node
  cannot guarantee that every workflow will fit, because it restrains model
  memory rather than setting a hard limit on all GPU memory use.
- H3 Sparse Attention enables fixed-density sparse attention while keeping text,
  reference conditioning, audio, non-video queries, and mixed boundary tiles
  dense. Its default video KV budget is 30 percent. The optional legacy
  early/late policy adds 30 percentage points to the first two and last two
  sampler steps, capped at 100 percent.
- H3 Sparse Attention (Advanced) exposes explicit early and late density
  windows plus a sparse-backend selector. Video KV budget controls the middle
  steps; Early steps/Early KV and Late steps/Late KV independently control the
  edges. The defaults are two early steps at 50 percent KV and two late steps at
  50 percent KV. If the two windows overlap, the denser of the two requested
  edge budgets is used. Backend `auto` uses native Kitchen INT8 at 64Q x 64KV,
  then an
  existing compatible Sparse Sage installation, INT8 Triton, FlexAttention,
  and finally the resolved dense H3 path. FlexAttention uses FP8 carriers on
  supported NVIDIA runtimes and native BF16/FP16 Q/K/V through Triton on ROCm.
  Explicit backend selections are hard requirements and error if unavailable;
  bypass the node to force dense attention.

The production native backend and the explicit quality arms execute 64Q x
64KV, 128Q x 128KV, or 128Q x 64KV routed geometry through the native INT8
kernel. They reuse the same
chunked Kitchen Q128 quantization carrier; the 64Q kernel consumes each 64-row
half directly rather than requantizing Q. Each geometry has a matched Sol arm.
The hard arm drops rejected tiles. The Sol arm approximates rejected 64Q x 64KV
tiles with block-mean K and block-sum V and merges that residual into the native
kernel's softmax state. In the 5 percent FL2VA baker and robot stress cases,
64Q x 64KV was dramatically more robust than either larger geometry. The
matched Sol residual did not visibly improve 128Q x 64KV or 64Q x 64KV and
added sampler time, so `auto` uses hard 64Q x 64KV and never selects Sol. The
explicit Sol choices remain available for reproducing experiments.

> **Sparse attention changes model computation. It is not free acceleration.**
> Lower Video KV budgets retain fewer target-video attention connections and can
> reduce prompt adherence, change motion/detail, or otherwise change the result.
> There is no attention percentage that is lossless for every prompt. The quality
> cost also depends on where attention is removed in the denoising schedule; H3
> is especially sensitive to reduced attention in the early sampling steps.

The production nodes are grouped under H3-Optimizations > Model Patches.

The production nodes are order-independent. Unsupported model families pass
through unchanged. Auto modes retain the existing implementation when a
specialized provider cannot satisfy its complete format and runtime contract.
A saved workflow containing either H3 Sparse Attention node first uses the
shipped native Kitchen backend when its per-GPU self-test passes. An existing
compatible Sparse Sage installation is next, followed by INT8 Triton,
FlexAttention, and the resolved dense H3 path. The selected fallback and reason
appear in the node status text. Explicit advanced early/middle/late budgets are
preserved across all sparse backends.

## Performance

The table below is a representative end-to-end sampler sweep comparing dense
Comfy attention against the optimization pack at two video lengths. Times are
median sampler-step times from the sweep. The 20-step columns are simple
projections from the measured step time, so they are useful for scale but are
not separate wall-clock measurements. These measurements predate the 64Q x
64KV production default and do not measure its current performance.

The 5-second VRAM measurements are deliberately omitted. That workload had
enough free memory that allocation/offload behavior was demand-driven and
run-order sensitive, making those peaks misleading. The 10-second workload put
the card under genuine memory pressure, so its peak figures are the useful
comparison from this sweep.

| configuration | 5s step | 5s 20-step | 10s step | 10s 20-step | 10s peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Comfy Kitchen INT8 dense (no pack) | 27.4 s | ~9m08s | 85.1 s | ~28m22s | 11800 MiB |
| SageAttention dense (no pack) | 27.5 s | ~9m09s | out of memory | - | - |
| H3 Memory Optimization | 25.5 s | ~8m29s | 75.1 s | ~25m02s | 9121 MiB |
| H3 Memory Optimization + H3 Sparse Attention (KV 100%) | 25.3 s | ~8m26s | 75.9 s | ~25m18s | 8763 MiB |
| **H3 Memory Optimization + H3 Sparse Attention (KV 30%, default)** | **15.7 s** | **~5m13s** | **39.2 s** | **~13m04s** | **9425 MiB** |

Against public dense Comfy Kitchen attention, the default 30 percent sparse
configuration measured **1.75x faster at 5 seconds** and **2.17x faster at 10
seconds**. The advantage increased with sequence length in this sweep rather
than remaining a fixed multiplier.

### Where the measured speedup comes from

The same sweep was arranged as an attribution ladder so each major change could
be measured independently:

| change | 5s | 10s |
| --- | ---: | ---: |
| chunked QKV and bounded MLP | +7.5% | +13.3% |
| public dense Kitchen to the native Kitchen INT8 kernel | +0.7% | -1.0% |
| video KV density 100% to 30% | +61.5% | +93.7% |

Sparsity is the dominant acceleration. Chunked QKV and bounded MLP provide a
smaller but real gain. The native Kitchen kernel at 100 percent KV density is
approximately performance-neutral against ComfyUI's public dense Kitchen path
in this measurement; its main value here is providing the execution path that
can consume the sparse route efficiently.

Sparse attention is therefore not free speed. `KV` density is a
quality/prompt-adherence budget as well as a performance setting, and its
visible effect depends on the prompt and on where in the diffusion schedule the
removed attention occurs. Some prompts tolerate very low budgets while others
require substantially more. The default is intended as a practical
speed/quality tradeoff, not a claim that 30 percent density is lossless.

These numbers are measurements from one test setup, not universal performance
claims. GPU architecture, checkpoint format, sequence length, ComfyUI version,
and backend availability can all change the result.

## Performance

MiniMax H3 text-to-video at 1376x768 (1.0 MP, 16:9) on an RTX 4070 12 GB,
`res_multistep`/`simple`, measured end to end through a real sampler. Times are
the median of three sampler steps after a discarded warmup step; the 20-step
column extrapolates from that median.

| configuration | 5s step | 5s 20-step | 10s step | 10s 20-step | 10s peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Comfy Kitchen INT8 dense (no pack) | 27.4 s | ~9m08s | 85.1 s | ~28m22s | 11800 MiB |
| SageAttention dense (no pack) | 27.5 s | ~9m09s | out of memory | — | — |
| H3 Memory Optimization | 25.5 s | ~8m29s | 75.1 s | ~25m02s | 9121 MiB |
| H3 Memory Optimization + H3 Sparse Attention (KV 100%) | 25.3 s | ~8m26s | 75.9 s | ~25m18s | 8763 MiB |
| **H3 Memory Optimization + H3 Sparse Attention (KV 30%, default)** | **15.7 s** | **~5m13s** | **39.2 s** | **~13m04s** | **9425 MiB** |

Against Comfy Kitchen INT8 dense, the default configuration is **1.75x faster at
5s and 2.17x at 10s**. The advantage grows with sequence length, so a single
speedup number understates it on long clips and overstates it on short ones.

The three pack rows form an observed ladder. The last two steps change one
thing; the first combines chunked QKV and bounded MLP:

| step | 5s | 10s |
| --- | ---: | ---: |
| chunked QKV and bounded MLP | +7.5% | +13.3% |
| public dense Kitchen to the native Kitchen INT8 kernel | +0.7% | -1.0% |
| video KV density 100% to 30% | +61.5% | +93.7% |

Chunked QKV is a memory optimization, not a compute-speed optimization. The
observed QKV timing difference comes from Comfy Kitchen selecting CUTLASS config
13 for the full large-sequence projection while 4K chunks select the faster
config 0; forcing config 0 on the full projection accounts for that QKV speed
change. A future dispatcher fix may therefore remove the timing difference,
while the lower peak memory from chunking remains. The first row must not be
read as QKV-only speed attribution.

Sparsity is nearly all of it. The native kernel is not itself faster than
ComfyUI's public dense `comfy_kitchen_int8` at equal density -- it is what makes
the density reduction possible, and at 10s it is marginally slower at full
density. Anyone weighing the pack for its kernel alone should read that row.

VRAM is reported for 10s only, and deliberately. At 5s the card has enough
headroom that physical VRAM is demand-driven: DynamicVRAM expands into whatever
is free, so the peaks there measure available memory rather than what a
configuration needs, and comparing them is misleading. At 10s the card is under
real pressure and the figures separate: Comfy Kitchen dense reaches 11800 of
12282 MiB, the pack configurations sit between 8763 and 9425 MiB, and
SageAttention cannot complete the run at all.

These are physical whole-GPU peaks from `nvidia-smi`, not allocator high-water
marks, so they answer "does this fit on the card" rather than "how much did the
allocator hold". The allocator-level figures elsewhere in this README are
measured separately and are not comparable to these.

Reproduce with `benchmarks/bench_attention_arms.py`, which drives a running
ComfyUI server over its prompt API. The SageAttention row additionally requires
a `sageattention` package and ComfyUI-KJNodes for the patch node; it is measured
here as plain dense SageAttention, not Sparge/`spas_sage_attn`.

## Install

The package is configured for publication as `h3-optimizations` in the Comfy
Registry and is intended for installation through ComfyUI-Manager on current
ComfyUI builds.

Manual installation remains available from the ComfyUI custom-nodes directory:

    git clone https://github.com/Zironic/H3-Optimizations

Restart ComfyUI after cloning. The nodes then appear under H3-Optimizations.

The repository contains the Windows x64 DLL and Linux x86-64 shared library.
When the native backend is first resolved, the local binary is loaded, its ABI
is checked, and a cached per-GPU self-test is run. Nothing is downloaded,
compiled, or installed during startup. If the binary is missing, cannot load,
or fails the self-test, the nodes still load and `auto` uses the remaining
fallback chain.

The shipped CUDA targets are SM80, SM89, and SM120 on Windows, with SM90a also
included on Linux. Sparse Sage remains an optional explicit backend and an
automatic fallback when a compatible `spas_sage_attn` package is already
installed; this pack no longer installs or repairs it.

The projected native Kitchen sparse route consumes attention in 4K query
slices and writes each output projection directly into the disposable
normalized block input. Its full-forward output is bit-identical to the prior
route. Non-projected fallback execution retains sequence-major output storage
and early carrier release. Q/K chunks are quantized where they already are
rather than being copied contiguous first. At the 124-frame production shape
with AIMDO restrained to zero blocks, streamed output reduced incremental peak
VRAM by 382 MiB at 0.93 percent sampler cost. FinalLayer chunking reduced it by
1549 MiB at 0.04 percent cost; its sampled output relative RMSE was 0.000154.

Sparse `auto` uses the shipped native block-sparse Kitchen backend. Compatible
ConvRot-256 TensorWise INT8 QKV uses the native chunked producer and feeds its
carrier directly into sparse attention without materializing full BF16 Q/K/V.
Dense execution continues to use ComfyUI's public `comfy_kitchen_int8`
attention backend when that backend is selected. Compatible QKV uses the
specialized INT8 producer; checkpoint-native FP8 and ordinary BF16/FP16 QKV can
use held FP8 projection when Comfy reports accelerated FP8 support. Unsupported
quantized QKV formats remain on native Comfy projection. Sparse Sage requires a
separately installed compatible `spas_sage_attn` build. All specialized paths
retain their complete format and runtime gates.

Missing dense capabilities return to upstream H3 QKV and normal Comfy
attention. If native Kitchen is unavailable, `auto` tries an existing
compatible Sparse Sage package and then the package INT8 Triton sparse backend
on NVIDIA compute capability 8.0 or newer. Triton consumes the same route and,
for compatible ConvRot-256 TensorWise INT8 checkpoints, uses 4K QKV chunks to
produce its INT8 carriers directly. If that backend is unavailable,
FlexAttention is next. On supported NVIDIA runtimes, Flex stores Q/K/V as
per-head-scaled E4M3, restores Q/K scale before softmax, converts the FP8 output
back to the original floating dtype, and consumes the same route. Hopper and
Blackwell request the FA4 backend when its CuTe package is installed; other
supported NVIDIA runtimes use the Triton Flex kernel. On ROCm, Flex keeps Q/K/V
in native BF16/FP16 and uses PyTorch FlexAttention's Triton lowering with the
same H3 sparse block mask. The ROCm path is validated on first execution; if the
installed PyTorch/ROCm/Triton stack cannot lower the kernel, `auto` retires that
sparse signature and returns to dense attention. If no sparse backend is
available, the resolved dense H3 path remains the final fallback. Explicit
backend selections in the Advanced node do not traverse this chain.

The node status reports the provider selected at plan time. Held FP8 QKV
projection can still decline at runtime if the effective patched weight cannot
satisfy its binding contract; in that case QKV returns to standard projection
without changing the already selected attention backend.

## Compatibility

- Current ComfyUI with MiniMax H3 support and the `comfy_api.latest` extension API
- Python 3.10 or newer
- Windows x64 or Linux x86-64 for the shipped native Kitchen binaries
- Any backend supported by ComfyUI's MiniMax H3 implementation for the final
  dense fallback
- NVIDIA SM80 or newer for the shipped native Kitchen default
- NVIDIA SM80 or newer with Triton for the next local sparse fallback
- An FP8-capable NVIDIA GPU with PyTorch FlexAttention for the NVIDIA Flex
  fallback when INT8 Triton is unavailable
- A ROCm-capable PyTorch build with FlexAttention/Triton for the AMD sparse Flex
  fallback; incompatible ROCm stacks fall back to dense on first validation
- NVIDIA CUDA SM80, SM86, SM87, SM89, SM90, or SM120 for Sparse Sage

Dense QKV eligibility follows the complete producer specification returned by
Comfy Kitchen; it is not gated on a particular compute capability. Sparse Sage
accepts the exact ABI exported by a compatible `spas_sage_attn` build. Its
chunked QKV producer is selected only when the active kernel's Q/K tiles, scale
layouts, V carrier, accumulator, summaries, and callables all match. A
mismatched QKV format uses standard sparse QKV. An unvalidated Sparse Sage
architecture uses the next backend only in `auto`; an explicit Sparse Sage
request errors instead.

Production node IDs are H3MemoryOptimization, H3AIMDOResidencyLimiter,
H3SparseAttention, and H3SparseAttentionAdvanced. H3-Extended is not required.

## Validation

CPU tests cover node schemas, AIMDO limiter arithmetic and load callbacks, plan
composition, backend classification, native shipping contracts, explicit
sparse-backend selection, chunk boundaries and RoPE slices, non-H3 no-op
behavior, sparse contract and route geometry, runtime step/layout publication,
explicit early/middle/late density schedules,
deterministic video-only ordering permutations and inverses, fixed-density
ordering metrics, and source isolation. GPU kernel validation is intentionally
separate because it requires the matching hardware and compiled backend
packages. Flex CPU contracts cover the NVIDIA FP8 carrier path, the ROCm native
BF16/FP16 path, explicit Triton/FA4 selection, and first-call dense fallback in
backend `auto`.

Run the CPU suite from the ComfyUI root:

    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v

The AIMDO residency behavior benchmark uses synthetic 1/2/3/4-page Comfy
linear weights and runs every limiter level in a fresh process. Unlike a raw
VBAR fault probe, it completes ComfyUI's temporary-buffer fallback, verifies
the linear output, checks pin cleanup, and reports persistent VBAR pages,
temporary cast buffers, AIMDO usage, and whole-device VRAM separately:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\.agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1 -- .\custom_nodes\H3-Optimizations\benchmarks\bench_aimdo_residency.py --i-understand-this-uses-gpu --output .\.agent\tmp\aimdo-residency.json

The three-way attention benchmark derives each carrier contract from the
current checkout and verifies identical sparse routes before timing:

    .\.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_sparse_backends.py --output sparse-backends.json --i-understand-this-uses-gpu

## Acknowledgements

- Thanks to [Pizzawookiee](https://github.com/Pizzawookiee) for the low-VRAM
  experiments in [PR #13](https://github.com/Zironic/H3-Optimizations/pull/13)
  and [PR #26](https://github.com/Zironic/H3-Optimizations/pull/26). Although
  neither PR was merged as-is, those experiments helped inform the streamed
  QKV and FinalLayer chunking work that later shipped.
- The sparse-attention work draws ideas from
  [MoBA](https://github.com/MoonshotAI/MoBA) and
  [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/).
- This project relies heavily on
  [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) for quantization,
  ConvRot execution, and the kernel foundations used by the native attention
  backend.