# H3 Optimizations

Standalone production optimization nodes for MiniMax H3 in ComfyUI.

This pack owns its native Kitchen INT8 attention kernels, chunked QKV producer,
sparse routing, and bounded MLP execution. 

## Nodes

- H3 Memory Optimization selects the resolved dense H3 attention path and
  applies compatible memory/execution providers. ConvRot INT8 QKV can project
  in 4K token chunks directly into Comfy Kitchen carriers. Native FP8 uses held
  chunked FP8 execution, and ordinary BF16/FP16 QKV/MLP weights may be converted
  to FP8 E4M3 when accelerated FP8 is available. Enable the advanced `Preserve
  precision` toggle to forbid new quantization: dense attention and QKV stay on
  the upstream Comfy path, while floating MLP weights remain floating and still
  use bounded token chunking. Checkpoint-native quantization remains native
  where a compatible bounded MLP provider exists. NVFP4 and unsupported
  quantized formats preserve upstream Comfy execution.
- H3 Sparse Attention enables fixed-density native Kitchen INT8 attention while
  keeping text, reference conditioning, audio, non-video queries, and mixed
  boundary tiles dense. Its default video KV budget is 30 percent. The optional
  legacy early/late policy adds 30 percentage points to the first two and last
  two sampler steps, capped at 100 percent.
- H3 Sparse Attention (Advanced) exposes explicit early and late density
  windows plus a sparse-backend selector. Video KV budget controls the middle
  steps; Early steps/Early KV and Late steps/Late KV independently control the
  edges. The defaults are two early steps at 50 percent KV and two late steps at
  50 percent KV. If the two windows overlap, the denser of the two requested
  edge budgets is used. Backend `auto` uses native Kitchen INT8, then an
  existing compatible Sparse Sage installation, INT8 Triton, FP8
  FlexAttention, and the dense fallback. Explicit backend selections are hard
  requirements and error if unavailable; bypass the node to force dense
  attention.
The production nodes are grouped under H3-Optimizations > Model Patches.

The production nodes are order-independent. Unsupported model families pass through
unchanged. Auto modes retain the existing implementation when a specialized
provider cannot satisfy its complete format and runtime contract. A saved
workflow containing either H3 Sparse Attention node first uses the shipped
native Kitchen backend when its per-GPU self-test passes. An existing compatible
Sparse Sage installation is next, followed by INT8 Triton, FP8 FlexAttention,
and the resolved dense H3 path. The selected fallback and reason appear in the
node status text. Explicit advanced early/middle/late budgets are preserved
across all sparse backends.

## Install

From the ComfyUI custom-nodes directory:

    git clone https://github.com/Zironic/H3-Optimizations

Restart ComfyUI after cloning. The nodes then appear under H3-Optimizations;
ComfyUI-Manager is not required.

The repository contains the Windows x64 DLL and Linux x86-64 shared library.
When the native backend is first resolved, the local binary is loaded, its ABI is checked, and a cached per-GPU self-test is run.
Nothing is downloaded, compiled, or installed during startup. If the binary is
missing, cannot load, or fails the self-test, the nodes still load and `auto`
uses the remaining fallback chain.

The shipped CUDA targets are SM80, SM89, and SM120 on Windows, with SM90a also
included on Linux. Sparse Sage remains an optional explicit backend and an
automatic fallback when a compatible `spas_sage_attn` package is already
installed; this pack no longer installs or repairs it.

The native Kitchen sparse route holds roughly 900 MiB less at its peak than
it used to, with bit-identical output. Attention writes its output
sequence-major and hands back the same logical tensor, so the flatten before
the output projection is a view rather than a second full-sequence copy; the
INT8 carriers and the route are released once the kernel is launched instead
of being held across that projection; and Q/K chunks are quantized where they
already are rather than being copied contiguous first. Measured on one real
block at sequence 54006 and 30 percent density: peak allocated 4875 to 3952
MiB, peak reserved 5536 to 4684 MiB, and slightly faster.

Sparse `auto` uses the shipped native block-sparse Kitchen backend. Compatible
ConvRot-256 TensorWise INT8 QKV uses the native chunked producer and feeds its
carrier directly into sparse attention without materializing full BF16 Q/K/V.
Dense execution continues to use ComfyUI's public `comfy_kitchen_int8`
attention backend when that backend is selected. Compatible QKV uses the
specialized INT8 producer;
checkpoint-native FP8 and ordinary BF16/FP16 QKV can use held FP8 projection
when Comfy reports accelerated FP8 support. Unsupported quantized QKV formats
remain on native Comfy projection. Sparse Sage requires a separately installed
compatible `spas_sage_attn` build. All specialized paths retain their complete
format and runtime gates.

Missing dense capabilities return to upstream H3 QKV and normal Comfy
attention. If native Kitchen is unavailable, `auto` tries an existing
compatible Sparse Sage package and then the package INT8 Triton sparse backend
on NVIDIA compute capability 8.0 or newer. Triton consumes the same route and,
for compatible ConvRot-256 TensorWise INT8 checkpoints, uses 4K QKV chunks to
produce its INT8 carriers directly. If that backend is unavailable, PyTorch
FP8 FlexAttention is next. Flex stores Q/K/V as per-head-scaled E4M3, restores
Q/K scale before softmax, converts the FP8 output back to the original floating
dtype, and consumes the same route. Hopper and Blackwell request the FA4
backend when its CuTe package is installed; other supported runtimes use the
Triton Flex kernel. If neither sparse fallback is available, the resolved dense
H3 path remains the final fallback. Explicit backend selections in the Advanced
node do not traverse this chain.

The node status reports the provider selected at plan time. Held FP8 QKV
projection can still decline at runtime if the effective patched weight cannot
satisfy its binding contract; in that case QKV returns to standard projection
without changing the already selected attention backend.

## Compatibility

- Current ComfyUI with MiniMax H3 support
- Python 3.10 or newer
- Any backend supported by ComfyUI's MiniMax H3 implementation for the final
  dense fallback
- NVIDIA SM80 or newer for the shipped native Kitchen default
- NVIDIA SM80 or newer with Triton for the next local sparse fallback
- An FP8-capable NVIDIA GPU with PyTorch FlexAttention for the next sparse
  fallback when INT8 Triton is unavailable
- NVIDIA CUDA SM80, SM86, SM87, SM89, SM90, or SM120 for Sparse Sage

Dense QKV eligibility follows the complete producer specification returned by
Comfy Kitchen; it is not gated on a particular compute capability. Sparse Sage
accepts the exact ABI exported by a compatible spas_sage_attn build. Its
chunked QKV producer is selected only when the active kernel's Q/K tiles, scale
layouts, V carrier, accumulator, summaries, and callables all match. A
mismatched QKV format uses standard sparse QKV. An unvalidated Sparse Sage
architecture uses the next backend only in `auto`; an explicit Sparse Sage
request errors instead.

Production node IDs are H3MemoryOptimization, H3SparseAttention, and
H3SparseAttentionAdvanced. H3-Extended is not required.

## Validation

CPU tests cover node schemas, plan composition, backend classification, native
shipping contracts, explicit sparse-backend selection, chunk boundaries and
RoPE slices, non-H3 no-op
behavior, sparse contract and route geometry, runtime step/layout publication,
explicit early/middle/late density schedules, deterministic video-only ordering
permutations and inverses, fixed-density ordering metrics, and source isolation.
GPU kernel validation is intentionally separate because it requires the matching hardware
and compiled backend packages. FP8 Flex CPU contracts cover all-FP8 carrier
layouts, explicit Triton/FA4 selection, and first-call dense fallback in
backend `auto`.

Run the CPU suite from the ComfyUI root:

    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v

The three-way attention benchmark derives each carrier contract from the
current checkout and verifies identical sparse routes before timing:

    .\.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_sparse_backends.py --output sparse-backends.json --i-understand-this-uses-gpu
