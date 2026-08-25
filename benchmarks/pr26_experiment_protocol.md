# PR 26 worthwhile-idea GPU protocol

Run from the ComfyUI root only after the card is idle. These experiments do
not change node registration or production defaults. Keep every JSON artifact;
do not infer a win from console timing alone.

```powershell
nvidia-smi
New-Item -ItemType Directory -Force .agent\tmp\h3-pr26-results | Out-Null
```

## 1. FinalLayer FP32 chunking

This is a synthetic, equivalent whole-FinalLayer comparison. Memory and timing
arms use fresh processes; parity covers scalar and per-token timestep selectors.

```powershell
.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_final_layer_chunking.py `
  --output .agent\tmp\h3-pr26-results\final-layer.json `
  --i-understand-this-uses-gpu
```

Accept only if both parity cases stay within the recorded limits and chunking
materially reduces peak allocated memory without an unacceptable median-time
regression. Compare `peak_reserved_bytes` too, but treat it as allocator-facing
rather than tensor-only evidence.

## 2. Native Q-only byte parity

Build the side-car; it cannot overwrite the shipped sparse-attention DLL.

```powershell
.\custom_nodes\H3-Optimizations\native\build_q_only.ps1
.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_q_only_quantizer.py `
  --output .agent\tmp\h3-pr26-results\q-only.json `
  --i-understand-this-uses-gpu
```

Both cases must report exact Q bytes and exact Q scales. They deliberately
cover the `full_K <= 256` rotation-4 route, the long-K rotation-128 route, and
ragged Q tails. The Q-only time is descriptive only; the coupled oracle also
performs K/V work and is not an equivalent speed boundary.

## 3. Query-sliced Kitchen output lifetime

Set `$checkpoint` to the H3 diffusion checkpoint. First grade parity in one
process, then measure each arm alone in a fresh process so reserved-memory
high-water marks are not inherited from another arm.

```powershell
$checkpoint = 'C:\path\to\h3-diffusion-model.safetensors'
.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_memory_experiments.py `
  --checkpoint $checkpoint `
  --variants baseline,recommended,stream_output `
  --output .agent\tmp\h3-pr26-results\stream-output-parity.json `
  --i-understand-this-uses-gpu

.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_memory_experiments.py `
  --checkpoint $checkpoint --variants recommended --no-parity `
  --output .agent\tmp\h3-pr26-results\recommended-alone.json `
  --i-understand-this-uses-gpu

.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_memory_experiments.py `
  --checkpoint $checkpoint --variants stream_output --no-parity `
  --output .agent\tmp\h3-pr26-results\stream-output-alone.json `
  --i-understand-this-uses-gpu
```

`stream_output` intentionally retains the full Q carrier. It isolates only
query-sliced sparse-attention output, immediate out-projection, and reuse of
the disposable normalized input. A useful result is exact real-block parity,
the required native Kitchen route in the artifact, and a meaningful enclosing
block peak reduction versus `recommended`.

To test whether two-pass V becomes useful only inside this schedule, first
build the existing V-staging side-car, then repeat parity/fresh-arm measurement
with `stream_output_two_pass_v`.

```powershell
.\custom_nodes\H3-Optimizations\native\build_v_staging.ps1
```

## 4. Stage-local prefetch residency

This uses Comfy's real AIMDO cast, VBAR fault, pin, cleanup, and streaming
paths with H3-sized synthetic weight groups. It tests the mechanism, not H3
end-to-end speed.

```powershell
.venv\Scripts\python.exe custom_nodes\H3-Optimizations\benchmarks\bench_stage_prefetch.py `
  --output .agent\tmp\h3-pr26-results\stage-prefetch.json `
  --i-understand-this-uses-gpu
```

Accept the mechanism only if both arms preserve exact linear outputs, the
five-page watermark is reported, stage-local cleanup leaves no pins, and
whole-device peak or streaming-fallback behavior improves. A positive result
still requires a later real-block or full-workflow timing before production
integration.
