'''Phase 1: attribute the chunked Kitchen QKV producer, arm against arm.

Two arms run over one real H3 DiT block, both on the shipped 30 percent
sparse Kitchen route with the bounded ConvRot MLP:

  A  full-sequence QKV, Kitchen packs the whole carrier      (chunking OFF)
  B  chunked Kitchen QKV producing the carrier directly      (chunking ON)

Both arms are driven through the production ``make_forward`` and the
production backends. Nothing here reimplements the producer, because a
benchmark-local copy of the path stops describing the path being shipped the
first time either one changes.

Forwards alternate A B B A A B ... in a single process, so the arms share
weights, clocks, allocator state and driver state. Every forward records its
own CUDA event pairs and the process synchronizes once, after the forward, so
no measured region contains a stall this harness introduced.

Outer regions are authoritative. ``attention_total`` is measured directly;
the producer figure the two arms are compared on is derived from it by
subtracting the regions after the carrier is finished, so unattributed work
stays inside the number rather than vanishing between children.
'''

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


PRODUCTION_ALLOC_CONF = (
    'backend:native,garbage_collection_threshold:0.95,expandable_segments:False'
)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', PRODUCTION_ALLOC_CONF)

PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = Path(__file__).resolve().parent
for _root in (str(BENCHMARK_ROOT), str(COMFY_ROOT), str(PACK_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from h3_stage_recorder import StageRecorder, percentile, summarize  # noqa: E402

DEFAULT_SEQUENCE = 54_006
DEFAULT_VIDEO_START = 256
DEFAULT_QKV_CHUNK_ROWS = 4_096

# Regions that run after the carrier and route are finished. Everything else
# inside attention_total belongs to the producer, whichever arm produced it.
AFTER_CARRIER = ('sparse_attention_kernel', 'attention_out')
ROUTE_REGIONS = ('sparse_route', 'sparse_carrier_prepare')


def balanced_schedule(count):
    '''A B B A A B ... - adjacent pairs are balanced, so slow drift cancels.'''
    pattern = ('A', 'B', 'B', 'A')
    return [pattern[index % 4] for index in range(count)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Attribute the chunked Kitchen QKV producer against the '
                    'full-sequence control on one real H3 block.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--video-start', type=int, default=DEFAULT_VIDEO_START)
    parser.add_argument('--video-budget', type=float, default=0.3)
    parser.add_argument(
        '--qkv-chunk-rows', type=int, default=DEFAULT_QKV_CHUNK_ROWS
    )
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument(
        '--forwards', type=int, default=24,
        help='total forwards across both arms; must be a multiple of four',
    )
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0:
        parser.error('--sequence must be positive')
    if not 0 < args.video_start < args.sequence:
        parser.error('--video-start must be inside the sequence')
    if not 0.01 <= args.video_budget <= 1.0:
        parser.error('--video-budget must be in [0.01, 1]')
    if args.qkv_chunk_rows <= 0 or args.qkv_chunk_rows % 128:
        parser.error('--qkv-chunk-rows must be a positive multiple of 128')
    if args.warmup < 0 or args.forwards <= 0 or args.forwards % 4:
        parser.error('--forwards must be a positive multiple of four')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU '
            'preflight'
        )
    return args


def derived(forward):
    '''Per-forward figures taken from outer regions, not from child sums.'''
    def total(name):
        record = forward.get(name)
        return 0.0 if record is None else float(record['gpu_ms'])

    attention_total = total('attention_total')
    after = sum(total(name) for name in AFTER_CARRIER)
    route = sum(total(name) for name in ROUTE_REGIONS)
    return {
        'attention_total': attention_total,
        'producer_with_route': max(0.0, attention_total - after),
        'producer_without_route': max(0.0, attention_total - after - route),
        'route': route,
        'sparse_attention_kernel': total('sparse_attention_kernel'),
        'attention_out': total('attention_out'),
    }


def spread(values):
    values = [float(value) for value in values]
    return {
        'median_ms': statistics.median(values),
        'p10_ms': percentile(values, 0.10),
        'p90_ms': percentile(values, 0.90),
        'min_ms': min(values),
        'max_ms': max(values),
        'iqr_ms': percentile(values, 0.75) - percentile(values, 0.25),
        'samples': len(values),
    }


def paired_delta(rows_a, rows_b, key):
    '''B minus A over the balanced pairing, so run drift cancels per pair.'''
    pairs = min(len(rows_a), len(rows_b))
    deltas = [
        float(rows_b[index][key]) - float(rows_a[index][key])
        for index in range(pairs)
    ]
    if not deltas:
        return None
    summary = spread(deltas)
    summary['pairs'] = len(deltas)
    return summary


def fingerprint(torch, x, sequence, hidden):
    rows = torch.tensor(
        (0, sequence // 2, sequence - 1), dtype=torch.int64, device=x.device
    )
    columns = torch.tensor(
        (0, 1, hidden // 2, hidden - 2, hidden - 1),
        dtype=torch.int64,
        device=x.device,
    )
    return (
        x.index_select(0, rows)
        .index_select(1, columns)
        .float()
        .cpu()
        .tolist()
    )


def report(result):
    print(
        'sequence %d, %d heads x %d, chunk %d rows (%d chunks), %s, %d forwards'
        % (
            result['sequence'],
            result['heads'],
            result['head_dim'],
            result['qkv_chunk_rows'],
            result['qkv_chunks_per_block'],
            result['gpu']['architecture'],
            result['arms']['A']['forwards'] + result['arms']['B']['forwards'],
        )
    )
    print('schedule %s' % result['schedule'])
    print()
    header = '%-30s %10s %10s %10s %8s %8s' % (
        'stage', 'A ms', 'B ms', 'B-A ms', 'A calls', 'B calls'
    )
    print(header)
    print('-' * len(header))
    names = sorted(
        set(result['arms']['A']['stages']) | set(result['arms']['B']['stages'])
    )
    for name in names:
        a = result['arms']['A']['stages'].get(name)
        b = result['arms']['B']['stages'].get(name)
        a_ms = None if a is None else a['median_gpu_ms']
        b_ms = None if b is None else b['median_gpu_ms']
        print('%-30s %10s %10s %10s %8s %8s' % (
            name,
            '-' if a_ms is None else '%.3f' % a_ms,
            '-' if b_ms is None else '%.3f' % b_ms,
            '-' if a_ms is None or b_ms is None else '%+.3f' % (b_ms - a_ms),
            '-' if a is None else '%g' % a['median_calls'],
            '-' if b is None else '%g' % b['median_calls'],
        ))
    print()
    print('derived from outer regions (authoritative):')
    print('-' * len(header))
    for key, delta in result['paired_delta_b_minus_a'].items():
        a = result['arms']['A']['derived'][key]
        b = result['arms']['B']['derived'][key]
        print(
            '%-30s %10.3f %10.3f %10s   paired %+.3f [p10 %+.3f, p90 %+.3f]'
            % (
                key,
                a['median_ms'],
                b['median_ms'],
                '%+.3f' % (b['median_ms'] - a['median_ms']),
                delta['median_ms'],
                delta['p10_ms'],
                delta['p90_ms'],
            )
        )
    print()
    for arm in ('A', 'B'):
        print('arm %s peak: %.1f MiB allocated / %.1f MiB reserved' % (
            arm,
            result['arms'][arm]['peak_allocated_bytes'] / 2 ** 20,
            result['arms'][arm]['peak_reserved_bytes'] / 2 ** 20,
        ))


def main(argv=None):
    args = parse_args(argv)

    import torch

    from bench_chunked_kitchen_qkv import make_rope, resolve_checkpoint
    from bench_h3_block import build_block
    from bench_h3_block_allocator import make_layout, mod_segments_from_layout
    from h3_optimizations import diagnostics
    from h3_optimizations.attention.sparse.config import (
        HybridSparseConfig,
        MODE_SAGE128,
        MODE_SAGE128_FUSED_QKV,
    )
    from h3_optimizations.attention.sparse.kitchen_sparse import (
        SparseKitchenBackend,
        preflight_sparse_kitchen,
    )
    from h3_optimizations.attention_forward import make_forward as make_attention
    from h3_optimizations.kitchen_qkv import (
        ChunkedKitchenQKVProjector,
        resolve_kitchen,
    )
    from h3_optimizations.memory.config import (
        DEFAULT_CHUNK_ROWS as MLP_CHUNK_ROWS,
        MODE_CONVROT_2SLICE,
        ActivationMemoryConfig,
    )
    from h3_optimizations.memory.forward import make_forward as make_block_forward
    from h3_optimizations.qkv.formats import describe_linear
    from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    if torch.cuda.get_allocator_backend() != 'native':
        raise SystemExit(
            'the active CUDA allocator is not the production native backend'
        )

    checkpoint = resolve_checkpoint(args.checkpoint)
    block, prefix, hidden, t_dim = build_block(
        torch, checkpoint, args.block, device
    )
    qkv_format = describe_linear(block.attn.qkv_proj)
    if not qkv_format.convrot_int8_256:
        raise SystemExit(
            'this experiment requires ConvRot-256 TensorWise INT8 QKV; '
            'checkpoint QKV is %s' % qkv_format.label
        )

    kitchen = preflight_sparse_kitchen(
        cuda_available=lambda: True,
        capability_getter=lambda: torch.cuda.get_device_capability(device),
    )
    producer_kitchen = resolve_kitchen(device)
    if producer_kitchen is None:
        raise SystemExit('the shipped Kitchen QKV producer is unavailable')
    shape = (1, int(block.attn.heads), args.sequence, int(block.attn.head_dim))
    spec = producer_kitchen.int8_attention_producer_spec(
        shape, shape, dtype=torch.bfloat16, device=device
    )
    alignment = int(spec.sequence_alignment)
    if args.qkv_chunk_rows % alignment:
        raise SystemExit(
            'QKV chunk rows %d do not satisfy producer alignment %d'
            % (args.qkv_chunk_rows, alignment)
        )

    control = SparseKitchenBackend(
        HybridSparseConfig(mode=MODE_SAGE128, video_budget=args.video_budget),
        kitchen=kitchen,
    )
    projector = ChunkedKitchenQKVProjector(
        chunk_rows=args.qkv_chunk_rows, routing_summaries=True
    )
    candidate = SparseKitchenBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128_FUSED_QKV, video_budget=args.video_budget
        ),
        kitchen=kitchen,
        projector=projector,
    )
    for name, backend in (('A', control), ('B', candidate)):
        status = backend.as_status()
        if status['sparse_architecture'] != 'comfy_kitchen_int8':
            raise SystemExit('arm %s is not on native Kitchen sparse' % name)
        if int(status['sparse_kv_tile']) != 128:
            raise SystemExit('arm %s is not on the 128-wide KV tile' % name)
        if abs(float(status['video_budget']) - args.video_budget) > 1e-9:
            raise SystemExit('arm %s has the wrong video budget' % name)
    if control.as_status()['fused_qkv']:
        raise SystemExit('arm A must run with chunked QKV disabled')
    if not candidate.as_status()['fused_qkv']:
        raise SystemExit('arm B must run with chunked QKV enabled')

    forwards = {
        'A': make_attention(block.attn, args.block, backend=control),
        'B': make_attention(
            block.attn, args.block, backend=candidate, projector=projector
        ),
    }
    block_forward = make_block_forward(
        block,
        args.block,
        ActivationMemoryConfig(
            mode=MODE_CONVROT_2SLICE,
            chunk_rows=MLP_CHUNK_ROWS,
            strict=True,
        ),
        original_forward=block.forward,
    )

    layout = make_layout(args.sequence, args.video_start)
    mod_segments = mod_segments_from_layout(layout)
    options = {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=0,
            step_index=0,
            total_steps=max(1, args.forwards),
            layout=layout,
            compute_dtype=torch.bfloat16,
            device=device,
        )
    }
    generator = torch.Generator(device=device).manual_seed(args.seed)
    reference = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    x = reference.clone()
    t_emb = torch.randn(
        (1, t_dim), generator=generator, dtype=torch.float32, device=device
    )
    rope = make_rope(torch, args.sequence, device)

    recorder = StageRecorder(torch, device)

    def one_forward(arm):
        block.attn.forward = forwards[arm]
        x.copy_(reference)
        return block_forward(
            x, t_emb, mod_segments, rope, transformer_options=options
        )

    with diagnostics.installed(recorder):
        # Both arms warm up before either is measured, so neither pays the
        # other's first-call cost.
        for _ in range(args.warmup):
            for arm in ('A', 'B'):
                result = one_forward(arm)
                torch.cuda.synchronize(device)
                del result

        fingerprints = {}
        for arm in ('A', 'B'):
            result = one_forward(arm)
            torch.cuda.synchronize(device)
            fingerprints[arm] = fingerprint(torch, x, args.sequence, hidden)
            del result

        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        schedule = balanced_schedule(args.forwards)
        records = {'A': [], 'B': []}
        peaks = {'A': [], 'B': []}
        recorder.enabled = True
        for arm in schedule:
            torch.cuda.reset_peak_memory_stats(device)
            recorder.begin_forward()
            result = one_forward(arm)
            torch.cuda.synchronize(device)
            records[arm].append(recorder.end_forward())
            peaks[arm].append({
                'peak_allocated_bytes': int(
                    torch.cuda.max_memory_allocated(device)
                ),
                'peak_reserved_bytes': int(
                    torch.cuda.max_memory_reserved(device)
                ),
            })
            del result
        recorder.enabled = False

    rows = {arm: [derived(forward) for forward in records[arm]] for arm in records}
    arms = {}
    for arm in ('A', 'B'):
        arms[arm] = {
            'stages': summarize(records[arm]),
            'derived': {
                key: spread([row[key] for row in rows[arm]])
                for key in rows[arm][0]
            },
            'peak_allocated_bytes': max(
                point['peak_allocated_bytes'] for point in peaks[arm]
            ),
            'peak_reserved_bytes': max(
                point['peak_reserved_bytes'] for point in peaks[arm]
            ),
            'forwards': len(rows[arm]),
            'route': (control if arm == 'A' else candidate).as_status(),
            'per_forward': rows[arm],
        }

    result = {
        'phase': 'phase1_producer_attribution',
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'checkpoint_qkv_format': qkv_format.label,
        'block': int(args.block),
        'sequence': int(args.sequence),
        'hidden': int(hidden),
        'heads': int(block.attn.heads),
        'head_dim': int(block.attn.head_dim),
        'video_start': int(args.video_start),
        'video_budget': float(args.video_budget),
        'qkv_chunk_rows': int(args.qkv_chunk_rows),
        'qkv_chunks_per_block': (
            args.sequence + args.qkv_chunk_rows - 1
        ) // args.qkv_chunk_rows,
        'producer_alignment': alignment,
        'mlp_chunk_rows': int(MLP_CHUNK_ROWS),
        'seed': int(args.seed),
        'schedule': ''.join(balanced_schedule(args.forwards)),
        'arms': arms,
        'paired_delta_b_minus_a': {
            key: paired_delta(rows['A'], rows['B'], key)
            for key in rows['A'][0]
        },
        'output_fingerprints': fingerprints,
        'kernel_route': {
            'q_tile': 128,
            'kv_tile': int(candidate.executor.kv_tile),
            'cta_k': int(spec.cta_k),
            'kernel_head_dim': int(spec.kernel_head_dim),
            'route_encoding': 'delta',
            'backend': str(spec.backend),
        },
        'allocator': {
            'backend': torch.cuda.get_allocator_backend(),
            'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
            'matches_production_conf': (
                os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
                == PRODUCTION_ALLOC_CONF
            ),
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
            'architecture': 'sm%d%d' % torch.cuda.get_device_capability(device),
            'total_memory_bytes': int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        },
        'torch_version': torch.__version__,
    }

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        if not output.parent.is_dir():
            raise SystemExit('output directory does not exist: %s' % output.parent)
        output.write_text(serialized + '\n', encoding='utf-8')
    if args.json:
        print(serialized)
    else:
        report(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
