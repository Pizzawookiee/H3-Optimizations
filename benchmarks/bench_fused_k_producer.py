from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
BENCH = PACK / "benchmarks"
for path in (BENCH, ROOT, PACK):
    sys.path.insert(0, str(path))

from bench_chunked_kitchen_qkv import (  # noqa: E402
    HEAD_DIM,
    build_attention,
    make_rope,
    resolve_checkpoint,
)
from bench_fused_q_producer import alternating_time, tensor_difference  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kitchen-source", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--block", type=int, default=24)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--global-sequence", type=int, default=43008)
    parser.add_argument("--start", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--force-anchor", action="store_true")
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_understand_this_uses_gpu:
        parser.error("pass --i-understand-this-uses-gpu after managed preflight")
    if args.sequence % 128 or args.start % 128:
        parser.error("sequence and start must be multiples of 128")
    if args.start + args.sequence > args.global_sequence:
        parser.error("chunk lies outside global sequence")
    return args


def bind_fused_kv(path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    function = library.h3_int8_fused_kv
    function.restype = ctypes.c_int
    function.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_size_t,
    )
    library.h3_int8_last_error.restype = ctypes.c_char_p
    return library, function


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(args.kitchen_source).resolve()))

    import torch
    import torch.nn.functional as F
    from comfy_kitchen.backends import cuda
    from h3_optimizations.native import quantize_int8_rowwise_convrot256
    from h3_optimizations.native import producer as native_producer
    from h3_optimizations.qkv.streamed import PROJECTION_NATIVE, create_held_qkv

    device = torch.device("cuda")
    module, hidden, prefix = build_attention(
        torch, resolve_checkpoint(args.checkpoint), args.block, 1e-6, device
    )
    heads = int(module.heads)
    inner = heads * HEAD_DIM
    generator = torch.Generator(device=device).manual_seed(1234)
    global_x = torch.randn(
        (args.global_sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    global_rope = make_rope(torch, args.global_sequence, device)
    x = global_x[args.start : args.start + args.sequence].contiguous()
    rope = global_rope[:, args.start : args.start + args.sequence].contiguous()
    freqs = rope[0, :, 0].contiguous()

    spec = native_producer.int8_attention_producer_spec(
        (1, heads, 1, HEAD_DIM),
        (1, heads, args.global_sequence, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        cta_k=64,
    )
    sample_rows = torch.tensor(
        spec.k_anchor_positions, dtype=torch.int64, device=device
    )
    held = create_held_qkv(module, global_x[:1], PROJECTION_NATIVE)
    held.__enter__()
    try:
        _q, samples, _v = held.project_rows(global_x, global_rope, sample_rows)
    finally:
        held.__exit__(None, None, None)
    anchor = native_producer.select_int8_attention_k_anchor(spec, samples)
    selected_anchor_heads = int((anchor.indices >= 0).sum().item())
    if args.force_anchor:
        anchor = native_producer.Int8AttentionKAnchor(
            values=samples[:, :, 0].contiguous(),
            indices=torch.zeros((1, heads), dtype=torch.int32, device=device),
        )
    producer = native_producer.create_int8_attention_producer(spec, anchor)

    weight = module.qkv_proj.weight
    qdata = weight._qdata
    weight_scale = weight._params.scale.to(torch.float32).reshape(-1)
    kv_weight = qdata[inner : 3 * inner].contiguous()
    kv_weight_scale = weight_scale[inner : 3 * inner].contiguous()
    norm = module.k_norm.weight.contiguous()

    act, act_scale = quantize_int8_rowwise_convrot256(x)
    combined_v = torch.empty(
        (args.sequence, inner), dtype=torch.bfloat16, device=device
    )
    retained_v = torch.empty(
        (1, heads, args.global_sequence, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    projected_kv = torch.empty(
        (args.sequence, 2 * inner), dtype=torch.bfloat16, device=device
    )
    candidate_k = torch.empty(
        (heads, args.global_sequence, HEAD_DIM), dtype=torch.int8, device=device
    )
    candidate_scale = torch.empty(
        (heads, ((args.global_sequence + 63) // 64) * 4),
        dtype=torch.float32,
        device=device,
    )
    candidate_summary = torch.empty(
        (heads, args.sequence // 64, HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    dummy_q = torch.zeros(
        (1, heads, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    stream = torch.cuda.current_stream(device).cuda_stream

    native_library, fused_kv = bind_fused_kv(args.library)

    def wrap(tensor):
        return cuda._wrap_for_dlpack(tensor)

    def quantize_activation():
        nonlocal act, act_scale
        act, act_scale = quantize_int8_rowwise_convrot256(x)

    def gemm(weight_data, scales, output):
        used = cuda._C.cutlass_int8_dequant_config(
            wrap(act),
            wrap(weight_data),
            wrap(act_scale),
            wrap(scales),
            wrap(output),
            cuda.DTYPE_TO_CODE[torch.bfloat16],
            0,
            stream,
        )
        if not used:
            raise RuntimeError("Kitchen declined Config 0")

    normalized = None
    baseline_summary = None

    def finish_k(projected):
        nonlocal normalized, baseline_summary
        shaped = projected.view(1, args.sequence, heads, HEAD_DIM)
        normalized = F.rms_norm(
            shaped,
            (HEAD_DIM,),
            weight=norm,
            eps=module.k_norm.eps,
        )
        cuda.apply_rope_split_half1_(normalized[..., :96], rope)
        k_hnd = normalized.transpose(1, 2)
        baseline_summary = k_hnd.reshape(
            1, heads, args.sequence // 64, 64, HEAD_DIM
        ).mean(dim=-2).squeeze(0)
        native_producer._quantize_qk_chunk(
            native_producer.loader.load(),
            producer,
            spec,
            dummy_q,
            k_hnd,
            0,
            args.start,
        )

    def baseline_kv():
        gemm(kv_weight, kv_weight_scale, projected_kv)
        finish_k(projected_kv[:, :inner])
        retain_v(projected_kv[:, inner:])

    def retain_v(projected):
        shaped = projected.view(args.sequence, heads, HEAD_DIM)
        retained_v[:, :, args.start : args.start + args.sequence].copy_(
            shaped.transpose(0, 1).unsqueeze(0)
        )

    def candidate_kv():
        status = fused_kv(
            act.data_ptr(),
            kv_weight.data_ptr(),
            act_scale.data_ptr(),
            kv_weight_scale.data_ptr(),
            norm.data_ptr(),
            freqs.data_ptr(),
            anchor.values.data_ptr(),
            anchor.indices.data_ptr(),
            candidate_summary.data_ptr(),
            combined_v.data_ptr(),
            candidate_k.data_ptr(),
            candidate_scale.data_ptr(),
            args.sequence,
            2 * inner,
            hidden,
            args.global_sequence,
            args.start,
            module.k_norm.eps,
            stream,
        )
        if status:
            error = native_library.h3_int8_last_error()
            raise RuntimeError(error.decode("utf-8", "replace"))
        retain_v(combined_v)

    quantize_activation()
    baseline_kv()
    candidate_kv()
    gemm(kv_weight, kv_weight_scale, projected_kv)
    torch.cuda.synchronize(device)

    scale_start = (args.start // 64) * 4
    scale_stop = scale_start + (args.sequence // 64) * 4
    baseline_k_slice = producer.k[0, :, args.start : args.start + args.sequence]
    baseline_scale_slice = producer.k_scale[0, :, scale_start:scale_stop]
    candidate_k_slice = candidate_k[:, args.start : args.start + args.sequence]
    candidate_scale_slice = candidate_scale[:, scale_start:scale_stop]
    parity = {
        "k_exact": bool(torch.equal(candidate_k_slice, baseline_k_slice)),
        "k_mismatch_fraction": float(
            (candidate_k_slice != baseline_k_slice).float().mean().item()
        ),
        "k_scale_exact": bool(
            torch.equal(candidate_scale_slice, baseline_scale_slice)
        ),
        "k_scale_max_abs": float(
            (candidate_scale_slice - baseline_scale_slice).abs().max().item()
        ),
        "summary": tensor_difference(
            torch, candidate_summary, baseline_summary
        ),
        "v": tensor_difference(
            torch, combined_v, projected_kv[:, inner:]
        ),
        "anchor_enabled_heads": int((anchor.indices >= 0).sum().item()),
        "selected_anchor_heads": selected_anchor_heads,
    }

    timings = {
        "prequantized_kv_producer": alternating_time(
            torch,
            (
                ("baseline", baseline_kv),
                ("fused_kv", candidate_kv),
            ),
            args.warmup,
            args.iterations,
            args.repeats,
            device,
        ),
        "including_activation_quant_kv": alternating_time(
            torch,
            (
                ("baseline", lambda: (quantize_activation(), baseline_kv())),
                (
                    "fused_kv",
                    lambda: (quantize_activation(), candidate_kv()),
                ),
            ),
            args.warmup,
            args.iterations,
            args.repeats,
            device,
        ),
    }

    summary = {}
    for boundary, cases in timings.items():
        summary[boundary] = {
            name: result["cuda"]["median_ms"] for name, result in cases.items()
        }
        if "baseline" in summary[boundary]:
            base = summary[boundary]["baseline"]
            summary[boundary]["speedup_vs_baseline_percent"] = {
                name: (base / value - 1.0) * 100.0
                for name, value in summary[boundary].items()
                if name not in ("baseline", "speedup_vs_baseline_percent")
            }

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "checkpoint_prefix": prefix,
                "block": args.block,
                "sequence": args.sequence,
                "global_sequence": args.global_sequence,
                "start": args.start,
                "library": str(Path(args.library).resolve()),
                "parity": parity,
                "timings": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
