# Comfy Kitchen compatibility baseline

Captured for Phase 0 before implementing Phase A1 and A2.

## Versions

- ComfyUI: `3f7535441e63554968b5ec897c2ff63791ebf760`
- H3-Optimizations: `3d612198ed5305e0eabcb2d7d31fb9429eabfec7`
- Installed `comfy-kitchen`: `0.2.31`
- Original CPU baseline: 35 tests passed with `CUDA_VISIBLE_DEVICES=-1`
- Post-A1/A2 CPU baseline: 42 tests passed with `CUDA_VISIBLE_DEVICES=-1`

No GPU, model, inference, or benchmark work was run.

## Live contracts

- `comfy.ldm.minimax.model.Attention.forward` projects QKV, applies Comfy
  Kitchen RMSNorm/RoPE, clones V, wraps transposed Q/K/V containers, and calls
  Comfy's selected optimized attention. The captured V container is
  storage-independent but non-contiguous in final HND layout.
- `comfy.ldm.modules.attention.get_attention_function` is the public backend
  registry lookup. `comfy_kitchen_int8` is registered when available.
- Comfy's Kitchen container path prequantizes already-created floating Q/K/V;
  Kitchen does not expose hidden-state-to-QKV projection through this API.
- `ModelPatcher.set_model_optimized_attention` installs the public override
  and preserves the selected backend's `container_function`.

## Pre-deletion SM89 policy inventory

- `h3_optimizations/dense_resolver.py` selected package dense Sage by compute
  capability before A2 replaced that policy with public Comfy selection.
- `h3_optimizations/dense_backend.py` rejects projected dense carriers outside
  SM89.
- `h3_optimizations/attention/sage_mem_eff.py` rejects the package dense Sage
  path outside SM89.
- `h3_optimizations/qkv/providers.py` gates fused QKV on SM89 and separately
  requires the sparse SM89 128Q x 64KV contract.
- `h3_optimizations/attention/sparse/backend.py` and
  `attention/sparse/sparse_sage.py` retain SM89 fused-selection gates.
- `attention/sm89_compat.py`, `attention/v_snapshot_compat.py`, the dense
  fused-QKV modules, and package dense Sage architecture modules remain
  deletion work for A3/B2. Genuine external ABI names containing `sm89` need
  separate classification before removal.

## Test matrix

| Contract | Coverage after A1/A2 | Remaining work |
| --- | --- | --- |
| Dense explicit override composition | `test_dense_selection.py` covers explicit override precedence and official backend ordering | Sparse/token-refiner composition remains A4 |
| V source probe | Exact, changed, and unavailable source cases covered | Remove the shim when upstream guarantees the layout |
| V layout and math | Contiguity, storage independence, and CPU numerical parity covered | Authorized real Kitchen H3 smoke remains C2 |
| Sparse standard/fused selection | Existing provider tests cover current SM89 standard/fused policy | Complete device/spec table and non-SM89 matching remain B1/B2/C1 |
| Device/spec matching | Existing sparse executor/provider checks are partial | Complete carrier, scale, summary, V, dtype, and device mismatch table remains C1 |
| UI compatibility | Existing schema snapshot covers node IDs and input order/options | A4 tooltip and final status wording remain |
| W4A8 fallback | Generic chunked path exists | Explicit W4A8 fallback test remains C1 |
| Removed imports | Source isolation test covers cross-pack and banned runtime calls | Dense deletion/import assertions remain A3/C1 |
