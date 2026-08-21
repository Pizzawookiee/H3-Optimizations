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
provider cannot satisfy its complete format and runtime contract.

## Install

From the ComfyUI custom-nodes directory:

    git clone https://github.com/Zironic/H3-Optimizations

The first ComfyUI startup automatically installs Sparse Sage before loading the
nodes. Supported Windows installations use matching upstream wheels with pinned
hashes. Linux x86-64 installations build a pinned SpargeAttention revision from
source when `git` and the CUDA `nvcc` compiler are available. Both paths use
`--no-deps`, so they cannot replace Torch or other ComfyUI packages. Set
`H3_OPTIMIZATIONS_SKIP_SPARSE_INSTALL=1` to disable this behavior.
The first Linux source build may add several minutes to startup.

Verified automatic wheels cover CUDA 12.4 with Torch 2.5.1, CUDA 12.6 with
Torch 2.6.0, CUDA 12.8 with Torch 2.7.1, 2.8.0, or 2.9 and newer, and CUDA 13.0
with Torch 2.9 or newer. The Linux build requires Torch 2.3 or newer and CUDA
12.0 or newer. Other platforms and combinations require a compatible manual
`spas_sage_attn` build.

Dense execution uses ComfyUI's public `comfy_kitchen_int8` attention backend.
Chunked dense QKV additionally requires a Comfy Kitchen release exposing its
external INT8-attention producer contract and ConvRot-256 TensorWise INT8 QKV
weights. Sparse Attention requires a compatible spas_sage_attn build; its
native-carrier fused QKV and the ConvRot MLP path additionally require Triton.
Missing dense capabilities return to upstream H3 QKV and normal Comfy
attention. Missing capabilities for explicitly requested Sparse Attention
produce a clear error.

spas_sage_attn is not a normal package dependency because its compiled backend
must match Torch and CUDA. The guarded pre-startup installer acts only when the
package is absent and either an exact verified wheel or the pinned Linux source
build is available.

## Compatibility

- Current ComfyUI with MiniMax H3 support
- Python 3.10 or newer
- NVIDIA CUDA for optimized attention

Dense QKV eligibility follows the complete producer specification returned by
Comfy Kitchen; it is not gated on a particular compute capability. Sparse Sage
accepts the exact ABI exported by a compatible spas_sage_attn build. Its fused
QKV producer is selected only when the active kernel's Q/K tiles, scale layouts,
V carrier, accumulator, summaries, and callables all match. A mismatched
architecture uses standard sparse QKV.

Node IDs are H3MemoryOptimization and H3SparseAttention. H3-Extended is not
required.

## Validation

CPU tests cover node schemas, plan composition, dense capability fallback,
chunk boundaries and RoPE slices, non-H3 no-op behavior, sparse contract and
route geometry, runtime step/layout publication, and source isolation. GPU
kernel validation is intentionally separate because it requires the matching
hardware and compiled backend packages.

Run the CPU suite from the ComfyUI root:

    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v
