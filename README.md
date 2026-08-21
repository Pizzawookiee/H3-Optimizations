# H3 Optimizations

Standalone production optimization nodes for MiniMax H3 in ComfyUI.

This pack owns its attention kernels, Sparse Sage routing, packed-layout
runtime, QKV providers, and bounded MLP implementation. It does not import or
depend on ComfyUI-H3-Extended.

## Nodes

- H3 Memory Optimization selects a compatible dense Sage backend, uses fused
  QKV projection when the checkpoint and runtime support it, and bounds MLP
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

Then install the attention backends appropriate for your Python, Torch, CUDA,
and GPU combination and restart ComfyUI.

Dense Sage execution uses the SageAttention package selected by ComfyUI.
Sparse Attention additionally requires a compatible spas_sage_attn build.
Fused QKV and ConvRot MLP paths require Triton, comfy-kitchen ConvRot-256
TensorWise INT8 weights, and the supported GPU architecture. Missing optional
capabilities either keep the existing dense path or produce a clear error when
Sparse Attention was explicitly requested.

SageAttention and spas_sage_attn are intentionally not declared as automatic
package dependencies. They contain architecture-specific compiled code, and an
incompatible wheel can replace or break the Torch installation used by ComfyUI.

## Compatibility

- Current ComfyUI with MiniMax H3 support
- Python 3.10 or newer
- NVIDIA CUDA for optimized attention

Prepared dense Sage backends cover SM80, SM86, SM89, SM90, SM120, and SM121.
Sparse Sage accepts the exact ABI exported by compatible spas_sage_attn builds
for SM80/86/87, SM89, SM90, and SM120. Fused QKV is selected only for SM89,
Triton, ConvRot-256 TensorWise INT8 weights, and the 128Q x 64KV sparse ABI.

Node IDs are H3MemoryOptimization and H3SparseAttention. H3-Extended is not
required.

## Validation

CPU tests cover node schemas, plan composition, non-H3 no-op behavior, sparse
route geometry, early/late budget resolution, runtime step/layout publication,
and source isolation. GPU kernel validation is intentionally separate because
it requires the matching hardware and compiled backend packages.

Run the CPU suite from the ComfyUI root:

    $env:CUDA_VISIBLE_DEVICES = '-1'
    .\.venv\Scripts\python.exe -m unittest discover -s custom_nodes\H3-Optimizations\tests -p 'test_*.py' -v
