'''Isolate chunking of MiniMax H3's FP32 FinalLayer island.

The memory arms run in fresh child processes so allocator residue from the
stock implementation cannot make the chunked reserved-memory result look
better or worse.  A separate parity child evaluates both implementations over
the same weights and inputs, including scalar and per-token timestep selectors.
'''

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


RESULT_PREFIX = 'H3_FINAL_LAYER_RESULT='
ARMS = ('stock', 'chunked')
PRODUCTION_ALLOC_CONF = (
    'backend:native,garbage_collection_threshold:0.95,expandable_segments:False'
)

PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
for _root in (str(COMFY_ROOT), str(PACK_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare stock and row-chunked H3 FinalLayer execution.'
    )
    parser.add_argument('--sequence', type=int, default=54_006)
    parser.add_argument('--video-rows', type=int, default=53_200)
    parser.add_argument('--hidden', type=int, default=5_376)
    parser.add_argument('--t-dim', type=int, default=2_688)
    parser.add_argument('--video-dim', type=int, default=96)
    parser.add_argument('--audio-dim', type=int, default=32)
    parser.add_argument('--chunk-rows', type=int, default=4_096)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-abs-error', type=float, default=5e-2)
    parser.add_argument('--max-relative-rmse', type=float, default=1e-5)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    parser.add_argument('--_child', choices=(*ARMS, 'parity'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.sequence <= 1 or not 0 < args.video_rows < args.sequence:
        parser.error('--video-rows must be inside --sequence')
    if min(args.hidden, args.t_dim, args.video_dim, args.audio_dim) <= 0:
        parser.error('model dimensions must be positive')
    if args.chunk_rows <= 0:
        parser.error('--chunk-rows must be positive')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('iteration arguments are invalid')
    if args.max_abs_error < 0 or args.max_relative_rmse < 0:
        parser.error('parity tolerances must be non-negative')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def _selector(selected, start, stop):
    return selected if selected.ndim == 1 else selected[start:stop]


def chunked_final_layer(layer, x, t_emb, video_seg, audio_seg, chunk_rows):
    '''Equivalent FinalLayer boundary with only the FP32 target rows chunked.'''
    import torch

    shift, scale = layer.adaln_proj(t_emb)

    def project(segment, output):
        first, last, row = segment
        pieces = []
        selected_scale = scale[row]
        selected_shift = shift[row]
        for start in range(first, last, int(chunk_rows)):
            stop = min(start + int(chunk_rows), last)
            local_start = start - first
            local_stop = stop - first
            normalized = layer.norm(x[start:stop])
            value = (
                normalized * (1.0 + _selector(selected_scale, local_start, local_stop))
                + _selector(selected_shift, local_start, local_stop)
            ).to(torch.float32)
            pieces.append(output(value))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    return project(video_seg, layer.video_out), project(audio_seg, layer.audio_out)


def _tensor_error(torch, reference, actual):
    delta = actual.float() - reference.float()
    rmse = delta.square().mean().sqrt()
    reference_rms = reference.float().square().mean().sqrt()
    return {
        'exact': bool(torch.equal(reference, actual)),
        'max_abs': float(delta.abs().max().item()),
        'rmse': float(rmse.item()),
        'relative_rmse': float((rmse / reference_rms.clamp_min(1e-12)).item()),
    }


def _build_case(args, torch, device):
    import comfy.ops
    from comfy.ldm.minimax.model import FinalLayer

    generator = torch.Generator(device=device).manual_seed(args.seed)
    layer = FinalLayer(
        args.hidden,
        args.t_dim,
        args.video_dim,
        args.audio_dim,
        1e-6,
        dtype=torch.bfloat16,
        adaln_dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.disable_weight_init,
    )
    for parameter in layer.parameters():
        parameter.data.normal_(generator=generator)
    x = torch.randn(
        args.sequence, args.hidden, dtype=torch.bfloat16,
        device=device, generator=generator,
    )
    t_emb = torch.randn(
        2, args.t_dim, dtype=torch.bfloat16, device=device, generator=generator,
    )
    video = (0, args.video_rows, 0)
    audio = (args.video_rows, args.sequence, 1)
    return layer, x, t_emb, video, audio


def _run_child(args):
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', PRODUCTION_ALLOC_CONF)
    import torch

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    layer, x, t_emb, video, audio = _build_case(args, torch, device)

    def run(arm, selectors=(video, audio)):
        vseg, aseg = selectors
        if arm == 'stock':
            return layer(x, t_emb, vseg, aseg)
        return chunked_final_layer(
            layer, x, t_emb, vseg, aseg, args.chunk_rows
        )

    if args._child == 'parity':
        results = {}
        selector_cases = {
            'scalar': (video, audio),
            'per_token': (
                (video[0], video[1], torch.zeros(video[1] - video[0], dtype=torch.long, device=device)),
                (audio[0], audio[1], torch.ones(audio[1] - audio[0], dtype=torch.long, device=device)),
            ),
        }
        for name, selectors in selector_cases.items():
            stock = run('stock', selectors)
            chunked = run('chunked', selectors)
            torch.cuda.synchronize(device)
            results[name] = {
                'video': _tensor_error(torch, stock[0], chunked[0]),
                'audio': _tensor_error(torch, stock[1], chunked[1]),
            }
        return {'arm': 'parity', 'cases': results}

    arm = args._child
    for _ in range(args.warmup):
        output = run(arm)
        torch.cuda.synchronize(device)
        del output
    torch.cuda.empty_cache()
    times = []
    peaks = []
    for _ in range(args.iterations):
        torch.cuda.reset_peak_memory_stats(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = run(arm)
        end.record()
        torch.cuda.synchronize(device)
        times.append(float(begin.elapsed_time(end)))
        peaks.append((
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        ))
        del output
    return {
        'arm': arm,
        'median_ms': statistics.median(times),
        'samples_ms': times,
        'peak_allocated_bytes': max(point[0] for point in peaks),
        'peak_reserved_bytes': max(point[1] for point in peaks),
    }


def _run_arm(args, arm):
    command = [
        sys.executable, str(Path(__file__).resolve()), '--_child', arm,
        '--sequence', str(args.sequence), '--video-rows', str(args.video_rows),
        '--hidden', str(args.hidden), '--t-dim', str(args.t_dim),
        '--video-dim', str(args.video_dim), '--audio-dim', str(args.audio_dim),
        '--chunk-rows', str(args.chunk_rows), '--warmup', str(args.warmup),
        '--iterations', str(args.iterations), '--seed', str(args.seed),
        '--max-abs-error', str(args.max_abs_error),
        '--max-relative-rmse', str(args.max_relative_rmse),
        '--i-understand-this-uses-gpu',
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    marker = next((
        line[len(RESULT_PREFIX):] for line in reversed(completed.stdout.splitlines())
        if line.startswith(RESULT_PREFIX)
    ), None)
    if completed.returncode or marker is None:
        raise RuntimeError(
            '%s child failed (%d)\nstdout:\n%s\nstderr:\n%s'
            % (arm, completed.returncode, completed.stdout, completed.stderr)
        )
    return json.loads(marker)


def main(argv=None):
    args = parse_args(argv)
    if args._child:
        result = _run_child(args)
        print(RESULT_PREFIX + json.dumps(result, separators=(',', ':')))
        parity_ok = all(
            output['max_abs'] <= args.max_abs_error
            and output['relative_rmse'] <= args.max_relative_rmse
            for case in result.get('cases', {}).values()
            for output in case.values()
        )
        return 0 if args._child != 'parity' or parity_ok else 1

    parity = _run_arm(args, 'parity')
    arms = {arm: _run_arm(args, arm) for arm in ARMS}
    result = {
        'schema': 1,
        'experiment': 'h3_final_layer_fp32_chunking',
        'sequence': args.sequence,
        'video_rows': args.video_rows,
        'audio_rows': args.sequence - args.video_rows,
        'hidden': args.hidden,
        'chunk_rows': args.chunk_rows,
        'parity_limits': {
            'max_abs_error': args.max_abs_error,
            'max_relative_rmse': args.max_relative_rmse,
        },
        'parity': parity['cases'],
        'arms': arms,
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + '\n', encoding='utf-8')
    print(serialized if args.json else (
        'stock %.3f ms %.1f MiB allocated; chunked %.3f ms %.1f MiB allocated; parity within limits'
        % (
            arms['stock']['median_ms'], arms['stock']['peak_allocated_bytes'] / 2 ** 20,
            arms['chunked']['median_ms'], arms['chunked']['peak_allocated_bytes'] / 2 ** 20,
        )
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
