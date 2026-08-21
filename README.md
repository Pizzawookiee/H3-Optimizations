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
  boundary tiles dense. Its optional early/late policy adds 30 percentage
  points to the first two and last two sampler steps, capped at 100 percent.

Both nodes are grouped under H3-Optimizations > Model Patches.

Both nodes are order-independent. Unsupported model families pass through
unchanged. Auto modes retain the existing implementation when a specialized
provider cannot satisfy its complete format and runtime contract. A saved
workflow containing H3 Sparse Attention also remains runnable when Sparse Sage
is unavailable: FP8-capable NVIDIA GPUs use the same fixed-density router
through PyTorch FlexAttention, and other unsupported combinations keep the
resolved dense H3 path. The selected fallback and reason appear in the node
status text.

## Install

From the ComfyUI custom-nodes directory:

    git clone https://github.com/Zironic/H3-Optimizations

Restart ComfyUI after cloning. The two nodes then appear under
H3-Optimizations > Model Patches; ComfyUI-Manager is not required.

Before node registration, startup checks the active Torch backend in a child
process. Supported Windows NVIDIA installations use matching upstream Sparse
Sage wheels with pinned hashes. Linux x86-64 NVIDIA installations build a
pinned SpargeAttention revision from source when `git` and the CUDA `nvcc`
compiler are available. Both paths use `--no-deps`, so they cannot replace
Torch or other ComfyUI packages. The first Linux source build may add several
minutes to startup.

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
ABI capabilities select PyTorch FlexAttention on an FP8-capable NVIDIA GPU.
Flex retains Q in its normal floating dtype, stores K/V as per-head-scaled
E4M3, and consumes the router's 128Q x 64KV block route directly. If that API,
Dynamo, CUDA, or FP8 compute is unavailable, the resolved dense H3 path remains
the final fallback. Errors raised after a validated sparse backend begins
execution remain hard errors instead of silently changing attention behavior
mid-run.

spas_sage_attn is not a normal package dependency because its compiled backend
must match Torch, CUDA, and the GPU architecture. The guarded pre-startup
installer installs a missing package or repairs a failed ABI validation only
when a verified replacement is available.

## Compatibility

- Current ComfyUI with MiniMax H3 support
- Python 3.10 or newer
- Any backend supported by ComfyUI's MiniMax H3 implementation for the final
  dense fallback
- NVIDIA SM89 or newer with PyTorch FlexAttention and FP8 compute for the
  sparse fallback when Sparse Sage is unavailable
- NVIDIA CUDA SM80, SM86, SM87, SM89, SM90, or SM120 for Sparse Sage

Dense QKV eligibility follows the complete producer specification returned by
Comfy Kitchen; it is not gated on a particular compute capability. Sparse Sage
accepts the exact ABI exported by a compatible spas_sage_attn build. Its
chunked QKV producer is selected only when the active kernel's Q/K tiles, scale
layouts, V carrier, accumulator, summaries, and callables all match. A
mismatched QKV format uses standard sparse QKV. An unvalidated Sparse Sage
architecture uses dense H3 attention.

Node IDs are H3MemoryOptimization and H3SparseAttention. H3-Extended is not
required.

## Validation

CPU tests cover node schemas, plan composition, backend classification, dense
and Sparse Sage capability fallback, stale-install repair selection, chunk
boundaries and RoPE slices, non-H3 no-op behavior, sparse contract and
route geometry, runtime step/layout publication, and source isolation. GPU
kernel validation is intentionally separate because it requires the matching
hardware and compiled backend packages.

Run the CPU suite from the ComfyUI root:

    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v
