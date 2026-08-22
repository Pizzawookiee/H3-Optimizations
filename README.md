# H3 Optimizations

Standalone production optimization nodes for MiniMax H3 in ComfyUI.

This pack owns Sparse Sage routing, its sparse native-carrier QKV path, the H3
adapter for Comfy Kitchen's dense INT8 carriers, and bounded MLP execution. It
does not import or depend on ComfyUI-H3-Extended.

## Nodes

- H3 Memory Optimization selects Comfy Kitchen dense INT8 attention and, when
  Kitchen exposes a compatible producer contract, projects ConvRot INT8 QKV in
  4K token chunks directly into Kitchen-owned carriers. It also bounds MLP
  activation memory with token chunking.
- H3 Sparse Attention enables fixed-density Sparse Sage attention while
  keeping text, reference conditioning, audio, non-video queries, and mixed
  boundary tiles dense. Its default video KV budget is 30 percent. The optional
  legacy early/late policy adds 30 percentage points to the first two and last
  two sampler steps, capped at 100 percent.
- H3 Sparse Attention (Advanced) exposes explicit early and late density
  windows. Video KV budget controls the middle steps; Early steps/Early KV and
  Late steps/Late KV independently control the edges. The defaults are two
  early steps at 50 percent KV and two late steps at 50 percent KV. If the two
  windows overlap, the denser of the two requested edge budgets is used.

All three nodes are grouped under H3-Optimizations > Model Patches.

The nodes are order-independent. Unsupported model families pass through
unchanged. Auto modes retain the existing implementation when a specialized
provider cannot satisfy its complete format and runtime contract. A saved
workflow containing either H3 Sparse Attention node also remains runnable when
Sparse Sage is unavailable: supported NVIDIA GPUs first use the package INT8
Triton sparse backend, then FP8 FlexAttention when available, before keeping the
resolved dense H3 path. The selected fallback and reason appear in the node
status text. Explicit advanced early/middle/late budgets are preserved across
all sparse fallback backends.

## Install

From the ComfyUI custom-nodes directory:

    git clone https://github.com/Zironic/H3-Optimizations

Restart ComfyUI after cloning. The three nodes then appear under
H3-Optimizations > Model Patches; ComfyUI-Manager is not required.

Before node registration, startup checks the active Torch backend in a child
process. Supported Windows NVIDIA installations use matching upstream Sparse
Sage wheels with pinned hashes. Linux x86-64 NVIDIA installations build a
pinned SpargeAttention revision from source when `git` and the CUDA `nvcc`
compiler are available. Both paths use `--no-deps`, so they cannot replace
Torch or other ComfyUI packages. The first Linux source build may add several
minutes to startup.

Linux source builds enable Ninja with half the detected logical CPU count as
the default worker pool and two `nvcc` threads per worker. Existing `MAX_JOBS`
or `NVCC_THREADS` environment values override those defaults. The installer
verifies the pinned Git commit and its expected build settings before applying
this local build-only patch.

An existing `spas_sage_attn` installation is left unchanged when its compiled
ABI validates for the active Torch, CUDA, and GPU. If it is stale, startup
reinstalls it only when a verified Windows wheel or the pinned Linux source
build matches the runtime. Set `H3_OPTIMIZATIONS_SKIP_SPARSE_INSTALL=1` to
disable all automatic Sparse Sage installation and repair.

Verified automatic wheels cover CUDA 12.4 with Torch 2.5.1, CUDA 12.6 with
Torch 2.6.0, CUDA 12.8 with Torch 2.7.1, 2.8.0, or 2.9 and newer, and CUDA 13.0
with Torch 2.9 or newer. The Linux build requires Torch 2.3 or newer and CUDA
12.0 or newer, `git`, and `nvcc` 12.0 or newer.

ROCm, MPS, XPU, CPU, future GPU architectures, NVIDIA installations without a
matching wheel/build toolchain, and failed Sparse Sage builds are left
untouched. The nodes still load and H3 Sparse Attention uses FP8 FlexAttention
when the active NVIDIA GPU supports it, then the resolved dense H3 attention
path otherwise. A manual `spas_sage_attn` build is needed only to obtain Sparse
Sage acceleration on an otherwise unsupported NVIDIA combination; it is not
required to run workflows containing these nodes.

Dense execution uses ComfyUI's public `comfy_kitchen_int8` attention backend.
Chunked dense QKV additionally requires a Comfy Kitchen release exposing its
external INT8-attention producer contract and ConvRot-256 TensorWise INT8 QKV
weights. Sparse Attention requires a compatible spas_sage_attn build; its
native-carrier 4K chunked QKV and the ConvRot MLP path additionally require
Triton.
Missing dense capabilities return to upstream H3 QKV and normal Comfy
attention. Missing Sparse Sage dependency, device, architecture, or compiled
ABI capabilities select the package INT8 Triton sparse backend on NVIDIA
compute capability 8.0 or newer. It consumes the same 128Q x 64KV route and,
for compatible ConvRot-256 TensorWise INT8 checkpoints, uses 4K QKV chunks to
produce its INT8 carriers directly. If that backend is unavailable, PyTorch
FP8 FlexAttention is next. Flex stores Q/K/V as per-head-scaled E4M3, restores
Q/K scale before softmax, converts the FP8 output back to the original floating
dtype, and consumes the same route. Hopper and Blackwell request the FA4
backend when its CuTe package is installed; other supported runtimes use the
Triton Flex kernel. If neither sparse fallback is available, the resolved dense
H3 path remains the final fallback. Errors raised after a validated sparse
backend begins execution remain hard errors instead of silently changing
attention behavior mid-run.

spas_sage_attn is not a normal package dependency because its compiled backend
must match Torch, CUDA, and the GPU architecture. The guarded pre-startup
installer installs a missing package or repairs a failed ABI validation only
when a verified replacement is available.

## Compatibility

- Current ComfyUI with MiniMax H3 support
- Python 3.10 or newer
- Any backend supported by ComfyUI's MiniMax H3 implementation for the final
  dense fallback
- NVIDIA SM80 or newer with Triton for the first sparse fallback when Sparse
  Sage is unavailable
- An FP8-capable NVIDIA GPU with PyTorch FlexAttention for the next sparse
  fallback when INT8 Triton is unavailable
- NVIDIA CUDA SM80, SM86, SM87, SM89, SM90, or SM120 for Sparse Sage

Dense QKV eligibility follows the complete producer specification returned by
Comfy Kitchen; it is not gated on a particular compute capability. Sparse Sage
accepts the exact ABI exported by a compatible spas_sage_attn build. Its
chunked QKV producer is selected only when the active kernel's Q/K tiles, scale
layouts, V carrier, accumulator, summaries, and callables all match. A
mismatched QKV format uses standard sparse QKV. An unvalidated Sparse Sage
architecture uses dense H3 attention.

Node IDs are H3MemoryOptimization, H3SparseAttention, and
H3SparseAttentionAdvanced. H3-Extended is not required.

## Validation

CPU tests cover node schemas, plan composition, backend classification, dense
and Sparse Sage capability fallback, stale-install repair selection, chunk
boundaries and RoPE slices, non-H3 no-op behavior, sparse contract and route
geometry, runtime step/layout publication, explicit early/middle/late density
schedules, and source isolation. GPU kernel validation is intentionally
separate because it requires the matching hardware and compiled backend
packages. FP8 Flex CPU contracts cover all-FP8 carrier layouts, explicit
Triton/FA4 selection, and first-call dense fallback.

Run the CPU suite from the ComfyUI root:

    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v

The three-way attention benchmark derives each carrier contract from the
current checkout and verifies identical sparse routes before timing:

    .\.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_sparse_backends.py --output sparse-backends.json --i-understand-this-uses-gpu
