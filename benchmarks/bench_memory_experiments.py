'''Price the attention-side memory experiments on one real H3 block.

Each variant is one opt-in change against the shipped 30 percent sparse
Kitchen route with the bounded ConvRot MLP. They are measured separately
because they target different moments, and the peak only moves when the
current peak moves:

  baseline      the shipped path
  nhd           sequence-major attention output storage (kills the reshape copy)
  release       drop the Kitchen carriers and route before out_proj
  nhd_release   both output-side fixes
  strided_qk    stop materializing every Q/K chunk before the pack kernel
  two_pass_v    produce the V carrier without a full-sequence BF16 V
  stream_output query-slice attention output/out_proj into disposable input
  stream_output_two_pass_v
                combine streamed output with two-pass V to expose interactions
  all           every variant above at once

Two things every variant must survive before its numbers mean anything: the
block output has to be bit-identical to the baseline, and the route has to be
the one that was asked for. A variant that silently fell back to another path
would otherwise report a very encouraging number about nothing.

Peak allocated is the honest figure for what the tensors cost. Peak reserved
is what the user actually sees, and it moves for allocator reasons that
tensor arithmetic does not predict -- so both are reported, alongside a
segment histogram for the record.
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

# backend options, projector options
VARIANTS = {
    'baseline': ({}, {}),
    'nhd': ({'output_layout': 'nhd'}, {}),
    'release': ({'release_carrier_before_out_proj': True}, {}),
    'nhd_release': (
        {'output_layout': 'nhd', 'release_carrier_before_out_proj': True}, {}
    ),
    'strided_qk': ({}, {'strided_qk_input': True}),
    'two_pass_v': ({}, {'v_mode': 'two_pass'}),
    # Everything that measured free: the two output-side fixes plus strided
    # Q/K. Deliberately excludes two_pass_v, which costs a second projection
    # and moved neither peak.
    'recommended': (
        {'output_layout': 'nhd', 'release_carrier_before_out_proj': True},
        {'strided_qk_input': True},
    ),
    'all': (
        {'output_layout': 'nhd', 'release_carrier_before_out_proj': True},
        {'strided_qk_input': True, 'v_mode': 'two_pass'},
    ),
    'stream_output': (
        {'output_layout': 'nhd'}, {'strided_qk_input': True}
    ),
    'stream_output_two_pass_v': (
        {'output_layout': 'nhd'},
        {'strided_qk_input': True, 'v_mode': 'two_pass'},
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Measure the attention-side memory experiments on one '
                    'real H3 block.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--variants',
        default='baseline,nhd,release,nhd_release,strided_qk,recommended,two_pass_v,all',
    )
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--video-start', type=int, default=DEFAULT_VIDEO_START)
    parser.add_argument('--video-budget', type=float, default=0.3)
    parser.add_argument(
        '--qkv-chunk-rows', type=int, default=DEFAULT_QKV_CHUNK_ROWS
    )
    parser.add_argument(
        '--query-chunk-rows', type=int, default=DEFAULT_QKV_CHUNK_ROWS,
        help='query rows per streamed attention/out_proj step',
    )
    parser.add_argument('--score-chunk-tiles', type=int, default=None)
    parser.add_argument(
        '--v-backend', choices=('native', 'torch_reference'), default=None,
        help='which V staging implementation to measure; the default takes '
             'the native kernels when they are built and refuses to silently '
             'substitute the reference',
    )
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--forwards', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output')
    parser.add_argument('--json', action='store_true')
    parser.add_argument(
        '--no-parity', action='store_true',
        help='measure one variant alone, without the baseline comparison. '
             'Peak reserved is a process-wide high-water mark, so it only '
             'means anything when a single variant has run in the process; '
             'grade parity in a separate combined run.',
    )
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.variants.split(',') if name.strip()]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        parser.error('unknown variants: %s' % ', '.join(unknown))
    if 'baseline' not in names and not args.no_parity:
        parser.error(
            'baseline is required; every parity check is against it. Pass '
            '--no-parity to measure one variant alone for its reserved figure.'
        )
    if args.no_parity and len(names) != 1:
        parser.error('--no-parity measures exactly one variant')
    args.variant_names = names
    if args.sequence <= 0:
        parser.error('--sequence must be positive')
    if not 0 < args.video_start < args.sequence:
        parser.error('--video-start must be inside the sequence')
    if not 0.01 <= args.video_budget <= 1.0:
        parser.error('--video-budget must be in [0.01, 1]')
    if args.qkv_chunk_rows <= 0 or args.qkv_chunk_rows % 128:
        parser.error('--qkv-chunk-rows must be a positive multiple of 128')
    if args.query_chunk_rows <= 0 or args.query_chunk_rows % 128:
        parser.error('--query-chunk-rows must be a positive multiple of 128')
    if args.warmup < 0 or args.forwards <= 0:
        parser.error('iteration arguments are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU '
            'preflight'
        )
    return args


def spread(values):
    values = [float(value) for value in values]
    return {
        'median_ms': statistics.median(values),
        'p10_ms': percentile(values, 0.10),
        'p90_ms': percentile(values, 0.90),
        'min_ms': min(values),
        'max_ms': max(values),
        'samples': len(values),
    }


def report(result):
    header = '%-13s %10s %10s %12s %12s %9s' % (
        'variant', 'block ms', 'attn ms', 'alloc MiB', 'reserved MiB', 'parity'
    )
    print(header)
    print('-' * len(header))
    for name, row in result['variants'].items():
        matches = row['output_matches_baseline']
        print('%-13s %10.3f %10.3f %12.1f %12.1f %9s' % (
            name,
            row['block']['median_ms'],
            row['attention_total']['median_ms'],
            row['peak_allocated_bytes'] / 2 ** 20,
            row['peak_reserved_bytes'] / 2 ** 20,
            'n/a' if matches is None else ('exact' if matches else 'DIFFERS'),
        ))
    if 'baseline' not in result['variants']:
        return
    baseline = result['variants']['baseline']
    print()
    print('%-13s %12s %12s %10s' % (
        'variant', 'alloc delta', 'resv delta', 'block delta'
    ))
    print('-' * 50)
    for name, row in result['variants'].items():
        if name == 'baseline':
            continue
        print('%-13s %+12.1f %+12.1f %+10.3f' % (
            name,
            (row['peak_allocated_bytes'] - baseline['peak_allocated_bytes'])
            / 2 ** 20,
            (row['peak_reserved_bytes'] - baseline['peak_reserved_bytes'])
            / 2 ** 20,
            row['block']['median_ms'] - baseline['block']['median_ms'],
        ))


def main(argv=None):
    args = parse_args(argv)

    import torch

    from bench_chunked_kitchen_qkv import make_rope, resolve_checkpoint
    from bench_h3_block import build_block
    from bench_h3_block_allocator import (
        make_layout,
        memory_stats,
        mod_segments_from_layout,
        segment_histogram,
    )
    from h3_optimizations import diagnostics
    from h3_optimizations.attention.sparse.config import (
        HybridSparseConfig,
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
    from h3_optimizations.native import v_staging
    from h3_optimizations.qkv.formats import describe_linear
    from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot
    from streamed_kitchen_output import CapturingProjector, StreamedOutputBackend

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
    if args.qkv_chunk_rows % int(spec.sequence_alignment):
        raise SystemExit(
            'QKV chunk rows %d do not satisfy producer alignment %d'
            % (args.qkv_chunk_rows, spec.sequence_alignment)
        )

    v_backend = args.v_backend or v_staging.available_backend()
    needs_v = any(
        VARIANTS[name][1].get('v_mode') == 'two_pass'
        for name in args.variant_names
    )
    if needs_v and v_backend == v_staging.BACKEND_TORCH:
        # The reference is an ulp off the kernel's fast-math reciprocal, so it
        # loses the block-output parity check on rounding ties alone. Grading a
        # variant that cannot pass would report a failure about the oracle.
        raise SystemExit(
            'two_pass_v needs the native V staging kernels: the Torch '
            'reference differs from the carrier on rounding ties and would '
            'fail the parity check for that reason alone. Build them with '
            'native/build_v_staging.ps1, or drop two_pass_v and all from '
            '--variants.'
        )

    def build(name):
        backend_options, projector_options = VARIANTS[name]
        base_projector = ChunkedKitchenQKVProjector(
            chunk_rows=args.qkv_chunk_rows,
            routing_summaries=True,
            v_backend=(
                v_backend
                if projector_options.get('v_mode') == 'two_pass'
                else None
            ),
            **projector_options,
        )
        base_backend = SparseKitchenBackend(
            HybridSparseConfig(
                mode=MODE_SAGE128_FUSED_QKV, video_budget=args.video_budget
            ),
            kitchen=kitchen,
            projector=base_projector,
            score_chunk_tiles=args.score_chunk_tiles,
            **backend_options,
        )
        if name.startswith('stream_output'):
            projector = CapturingProjector(base_projector)
            backend = StreamedOutputBackend(base_backend, args.query_chunk_rows)
        else:
            projector = base_projector
            backend = base_backend
        status = backend.as_status()
        if status['sparse_architecture'] != 'comfy_kitchen_int8':
            raise SystemExit('%s is not on native Kitchen sparse' % name)
        if not status['fused_qkv'] or int(status['sparse_kv_tile']) != 128:
            raise SystemExit('%s did not reach the required route' % name)
        return backend, projector, status

    built = {name: build(name) for name in args.variant_names}
    forwards = {
        name: make_attention(
            block.attn, args.block, backend=backend, projector=projector
        )
        for name, (backend, projector, _) in built.items()
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
    reference_x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    x = reference_x.clone()
    t_emb = torch.randn(
        (1, t_dim), generator=generator, dtype=torch.float32, device=device
    )
    rope = make_rope(torch, args.sequence, device)
    recorder = StageRecorder(torch, device)

    def one_forward(name):
        block.attn.forward = forwards[name]
        x.copy_(reference_x)
        return block_forward(
            x, t_emb, mod_segments, rope, transformer_options=options
        )

    # Parity first, on a fresh forward per variant, before any timing.
    with diagnostics.installed(recorder):
        for name in args.variant_names:
            for _ in range(max(1, args.warmup)):
                result = one_forward(name)
                torch.cuda.synchronize(device)
                del result

        if args.no_parity:
            parity = {name: None for name in args.variant_names}
            differences = dict(parity)
        else:
            outputs = {}
            for name in args.variant_names:
                result = one_forward(name)
                torch.cuda.synchronize(device)
                outputs[name] = x.clone()
                del result
            parity = {
                name: bool(torch.equal(outputs[name], outputs['baseline']))
                for name in args.variant_names
            }
            differences = {
                name: (
                    None
                    if parity[name]
                    else float(
                        (outputs[name].float() - outputs['baseline'].float())
                        .abs()
                        .max()
                        .item()
                    )
                )
                for name in args.variant_names
            }
            del outputs
        torch.cuda.empty_cache()

        measured = {}
        for name in args.variant_names:
            rows = []
            peaks = []
            recorder.reset()
            recorder.enabled = True
            for _ in range(args.forwards):
                torch.cuda.reset_peak_memory_stats(device)
                recorder.begin_forward()
                with recorder.stage('block'):
                    result = one_forward(name)
                torch.cuda.synchronize(device)
                rows.append(recorder.end_forward())
                peaks.append(
                    (
                        int(torch.cuda.max_memory_allocated(device)),
                        int(torch.cuda.max_memory_reserved(device)),
                    )
                )
                del result
            recorder.enabled = False
            stages = summarize(rows)
            measured[name] = {
                'stages': stages,
                'block': spread([row['block']['gpu_ms'] for row in rows]),
                'attention_total': spread(
                    [row['attention_total']['gpu_ms'] for row in rows]
                ),
                'peak_allocated_bytes': max(point[0] for point in peaks),
                'peak_reserved_bytes': max(point[1] for point in peaks),
                'output_matches_baseline': parity[name],
                'max_abs_difference': differences[name],
                'route': built[name][2],
                'projector': built[name][1].installation_signature,
                'segments': segment_histogram(torch),
                'memory_stats': memory_stats(torch, device),
            }

    result = {
        'phase': 'memory_experiments',
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
        'query_chunk_rows': int(args.query_chunk_rows),
        'score_chunk_tiles': args.score_chunk_tiles,
        'mlp_chunk_rows': int(MLP_CHUNK_ROWS),
        'seed': int(args.seed),
        'forwards': int(args.forwards),
        'v_staging': {
            'backend': v_backend,
            'library': v_staging.native_library_path(),
        },
        'kernel_route': {
            'q_tile': 128,
            'kv_tile': 128,
            'cta_k': int(spec.cta_k),
            'kernel_head_dim': int(spec.kernel_head_dim),
            'route_encoding': 'delta',
        },
        'variants': measured,
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

    serialized = json.dumps(result, indent=2, sort_keys=True, default=repr)
    if args.output:
        destination = Path(args.output).resolve()
        if not destination.parent.is_dir():
            raise SystemExit(
                'output directory does not exist: %s' % destination.parent
            )
        destination.write_text(serialized + '\n', encoding='utf-8')
    if args.json:
        print(serialized)
    else:
        report(result)
    failures = [
        name for name, row in measured.items()
        if row['output_matches_baseline'] is False
    ]
    if failures:
        print(
            '\nFAIL: %s did not reproduce the baseline block output exactly'
            % ', '.join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
