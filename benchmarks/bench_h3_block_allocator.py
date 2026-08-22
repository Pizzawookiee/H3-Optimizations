'''Measure CUDA allocator block reuse between H3 QKV chunks and the bounded MLP.

One process measures exactly one (QKV rows, MLP rows) arm through the real
production block lifecycle: AdaLN, chunked Sparse Sage QKV, real sparse
attention, residual, attention release, ConvRot two-slice MLP, residual.

Unlike the timing benchmarks in this directory this script never calls
``torch.cuda.empty_cache()`` inside the measured series. The cached allocator
state between consecutive block forwards is the object of study, so clearing it
would destroy the measurement.

The primary observable is an address identity: whether the BF16 tensor that the
first full-width MLP ``fc1`` chunk allocates starts at an address that a QKV
projection chunk freed earlier in the same forward. Allocator statistics and
segment histograms are the quantitative secondary evidence.

Run one arm per process; use ``run_qkv_allocator_sweep.py`` to drive a sweep.
'''

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace


# Comfy2 launches its server with --disable-cuda-malloc, which this checkout's
# cuda_malloc.py turns into exactly this allocator configuration. Reproduce it
# before torch is imported so the measured allocator matches production.
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

from bench_h3_block import build_block  # noqa: E402
from bench_chunked_kitchen_qkv import make_rope, resolve_checkpoint  # noqa: E402


DEFAULT_SEQUENCE = 54006
DEFAULT_VIDEO_START = 256
DEFAULT_QKV_CHUNK_ROWS = 4096
DEFAULT_MLP_CHUNK_ROWS = 4096

# Modulation rows used by the H3 packed layout, mirroring bench_h3_block.
MOD_ROW_VIDEO = 0
MOD_ROW_TEXT = 1
MOD_ROW_AUDIO = 2

STAT_KEYS = (
    'allocated_bytes.all.current',
    'allocated_bytes.all.peak',
    'requested_bytes.all.current',
    'requested_bytes.all.peak',
    'reserved_bytes.all.current',
    'reserved_bytes.all.peak',
    'active_bytes.all.peak',
    'inactive_split_bytes.all.current',
    'inactive_split_bytes.all.peak',
    'inactive_split.all.current',
    'segment.all.current',
    'segment.all.peak',
    'segment.large_pool.current',
    'segment.large_pool.peak',
    'allocation.all.current',
    'num_alloc_retries',
    'num_ooms',
)

# Per-forward series keys, kept short so a 20-forward run stays readable.
SERIES_KEYS = (
    'reserved_bytes.all.current',
    'inactive_split_bytes.all.current',
    'inactive_split.all.current',
    'segment.large_pool.current',
    'num_alloc_retries',
)


# ---------------------------------------------------------------------------
# A model of the PyTorch native caching allocator.
#
# These constants mirror c10/cuda/CUDACachingAllocator.cpp for PyTorch 2.x.
# They are a *prediction*, reported next to the measurement so the two can be
# compared. Never treat them as an API; if a prediction and a measurement
# disagree, the measurement wins and this model is what needs fixing.
# ---------------------------------------------------------------------------

MIN_BLOCK_SIZE = 512
SMALL_SIZE = 1_048_576
SMALL_BUFFER = 2_097_152
LARGE_BUFFER = 20_971_520
MIN_LARGE_ALLOC = 10_485_760
ROUND_LARGE = 2_097_152


def round_request(nbytes):
    '''Round an allocation request the way the native allocator does.'''
    nbytes = int(nbytes)
    if nbytes <= MIN_BLOCK_SIZE:
        return MIN_BLOCK_SIZE
    return MIN_BLOCK_SIZE * ((nbytes + MIN_BLOCK_SIZE - 1) // MIN_BLOCK_SIZE)


def segment_size(nbytes):
    '''Predict the cudaMalloc segment the allocator creates for a request.'''
    size = round_request(nbytes)
    if size <= SMALL_SIZE:
        return SMALL_BUFFER
    if size < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    return ROUND_LARGE * ((size + ROUND_LARGE - 1) // ROUND_LARGE)


def splits_large_block(block_bytes, request_bytes):
    '''Predict whether serving a request from a cached large block splits it.'''
    return int(block_bytes) - round_request(request_bytes) > SMALL_SIZE


def predict_reuse(qkv_rows, mlp_rows, qkv_width, mlp_tile_width, item_size):
    qkv_request = int(qkv_rows) * int(qkv_width) * int(item_size)
    mlp_request = int(mlp_rows) * int(mlp_tile_width) * int(item_size)
    qkv_segment = segment_size(qkv_request)
    mlp_segment = segment_size(mlp_request)
    fits = qkv_segment >= round_request(mlp_request)
    remainder = qkv_segment - round_request(mlp_request) if fits else None
    if not fits:
        verdict = 'too_small'
    elif remainder == 0:
        verdict = 'exact_fit'
    elif not splits_large_block(qkv_segment, mlp_request):
        verdict = 'fit_without_split'
    else:
        verdict = 'fit_with_split'
    return {
        'qkv_request_bytes': qkv_request,
        'qkv_segment_bytes': qkv_segment,
        'mlp_request_bytes': mlp_request,
        'mlp_segment_bytes': mlp_segment,
        'mlp_fits_in_qkv_segment': bool(fits),
        'split_remainder_bytes': remainder,
        'verdict': verdict,
    }


# ---------------------------------------------------------------------------
# Address tracing
# ---------------------------------------------------------------------------


class AllocationTrace:
    '''Record allocation addresses without retaining any tensor.'''

    def __init__(self, qkv_width, mlp_tile_width):
        self.qkv_width = int(qkv_width)
        self.mlp_tile_width = int(mlp_tile_width)
        self.qkv = []
        self.fc1 = []
        self.fc2 = []
        self.enabled = False

    def reset(self):
        self.qkv.clear()
        self.fc1.clear()
        self.fc2.clear()

    @staticmethod
    def _entry(tensor):
        return (
            int(tensor.data_ptr()),
            int(tensor.numel()) * int(tensor.element_size()),
            int(tensor.shape[-2]) if tensor.ndim >= 2 else 0,
        )

    def record_qkv(self, tensor):
        if self.enabled:
            self.qkv.append(self._entry(tensor))

    def record_convrot(self, tensor):
        if not self.enabled:
            return
        width = int(tensor.shape[-1])
        if width == self.mlp_tile_width:
            self.fc1.append(self._entry(tensor))
        else:
            self.fc2.append(self._entry(tensor))


def install_tracing(block, trace, memory_linear):
    '''Wrap the two allocations under test. Returns an uninstall callable.'''
    qkv_proj = block.attn.qkv_proj
    original_qkv_forward = qkv_proj.forward
    original_convrot = memory_linear._convrot_linear

    def traced_qkv(*args, **kwargs):
        out = original_qkv_forward(*args, **kwargs)
        trace.record_qkv(out)
        return out

    def traced_convrot(x, qdata, scale, input_act=None):
        out = original_convrot(x, qdata, scale, input_act=input_act)
        trace.record_convrot(out)
        return out

    # ConvRotTwoSliceMLP resolves _convrot_linear from module globals in its
    # __init__, and memory/forward.py builds a fresh one per block forward, so
    # patching the module attribute reaches every later forward.
    qkv_proj.forward = traced_qkv
    memory_linear._convrot_linear = traced_convrot

    def uninstall():
        # forward was a bound class method, not an instance attribute.
        del qkv_proj.forward
        memory_linear._convrot_linear = original_convrot

    return uninstall


def address_histogram(entries):
    counts = {}
    for address, nbytes, rows in entries:
        key = '0x%x' % address
        record = counts.setdefault(
            key,
            {'bytes': nbytes, 'rows': rows, 'count': 0},
        )
        record['count'] += 1
        record['bytes'] = max(record['bytes'], nbytes)
    return counts


def summarize_trace(trace, qkv_request_bytes, mlp_request_bytes):
    '''Reduce one forward's addresses to the reuse question being asked.'''
    qkv_addresses = {address for address, _bytes, _rows in trace.qkv}
    full_qkv = [entry for entry in trace.qkv if entry[1] == qkv_request_bytes]
    full_fc1 = [entry for entry in trace.fc1 if entry[1] == mlp_request_bytes]
    first_full_fc1 = full_fc1[0] if full_fc1 else None
    return {
        'qkv_calls': len(trace.qkv),
        'qkv_full_width_calls': len(full_qkv),
        'qkv_distinct_addresses': len(qkv_addresses),
        'fc1_calls': len(trace.fc1),
        'fc1_full_width_calls': len(full_fc1),
        'fc1_distinct_addresses': len({a for a, _b, _r in trace.fc1}),
        'first_full_fc1_address': (
            None if first_full_fc1 is None else '0x%x' % first_full_fc1[0]
        ),
        # The headline result: did the first full MLP expansion land on an
        # address a QKV projection chunk had already used and released?
        'first_full_fc1_reuses_qkv_address': (
            None
            if first_full_fc1 is None
            else bool(first_full_fc1[0] in qkv_addresses)
        ),
        'any_fc1_reuses_qkv_address': bool(
            {address for address, _b, _r in trace.fc1} & qkv_addresses
        ),
    }


# ---------------------------------------------------------------------------
# Allocator statistics and segments
# ---------------------------------------------------------------------------


def memory_stats(torch, device):
    stats = torch.cuda.memory_stats(device)
    return {key: int(stats[key]) for key in STAT_KEYS if key in stats}


def series_point(torch, device, elapsed_ms):
    stats = torch.cuda.memory_stats(device)
    point = {key: int(stats[key]) for key in SERIES_KEYS if key in stats}
    point['elapsed_ms'] = float(elapsed_ms)
    return point


def segment_histogram(torch, limit=12):
    '''Group live cudaMalloc segments by size, largest total footprint first.'''
    grouped = {}
    for segment in torch.cuda.memory_snapshot():
        if segment.get('segment_type') != 'large':
            continue
        key = int(segment['total_size'])
        record = grouped.setdefault(
            key,
            {
                'total_size_bytes': key,
                'segments': 0,
                'allocated_bytes': 0,
                'active_bytes': 0,
                'blocks': 0,
                'inactive_blocks': 0,
                'inactive_bytes': 0,
            },
        )
        record['segments'] += 1
        record['allocated_bytes'] += int(segment.get('allocated_size', 0))
        record['active_bytes'] += int(segment.get('active_size', 0))
        for block in segment.get('blocks', ()):
            record['blocks'] += 1
            if block.get('state') != 'active_allocated':
                record['inactive_blocks'] += 1
                record['inactive_bytes'] += int(block.get('size', 0))
    ordered = sorted(
        grouped.values(),
        key=lambda record: record['total_size_bytes'] * record['segments'],
        reverse=True,
    )
    return ordered[:limit]


def segments_of_size(torch, sizes):
    '''Report the block layout of every segment whose size is under test.'''
    wanted = {int(size) for size in sizes if size}
    found = []
    for segment in torch.cuda.memory_snapshot():
        if int(segment.get('total_size', 0)) not in wanted:
            continue
        found.append(
            {
                'total_size_bytes': int(segment['total_size']),
                'allocated_bytes': int(segment.get('allocated_size', 0)),
                'blocks': [
                    {
                        'size_bytes': int(block.get('size', 0)),
                        'requested_bytes': int(block.get('requested_size', 0)),
                        'state': block.get('state'),
                    }
                    for block in segment.get('blocks', ())
                ],
            }
        )
    return found


# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------


def make_layout(sequence, video_start):
    '''Build the packed layout the Sparse Sage router expects.'''
    sequence = int(sequence)
    video_start = min(int(video_start), sequence - 1)
    text_stop = min(128, video_start)
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=(
            (0, text_stop, 'text'),
            (text_stop, video_start, 'audio'),
            (video_start, sequence, 'video'),
        ),
        video_shape=(1, 1, sequence - video_start),
        audio_t=max(0, (video_start - text_stop) // 2),
    )


def mod_segments_from_layout(layout):
    rows = {
        'text': MOD_ROW_TEXT,
        'audio': MOD_ROW_AUDIO,
        'video': MOD_ROW_VIDEO,
    }
    return tuple(
        (int(start), int(stop), rows[name])
        for start, stop, name in layout.segments
        if stop > start
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Measure CUDA allocator block reuse between chunked H3 QKV '
            'projection and the bounded ConvRot two-slice MLP.'
        )
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--video-start', type=int, default=DEFAULT_VIDEO_START)
    parser.add_argument('--video-budget', type=float, default=0.5)
    parser.add_argument(
        '--qkv-chunk-rows',
        type=int,
        default=DEFAULT_QKV_CHUNK_ROWS,
        help='rows per chunked Sparse Sage QKV projection (multiple of 128)',
    )
    parser.add_argument(
        '--mlp-chunk-rows',
        type=int,
        default=DEFAULT_MLP_CHUNK_ROWS,
        help='rows per bounded MLP token slab (multiple of 256)',
    )
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--forwards', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--label', default='')
    parser.add_argument(
        '--record-history',
        action='store_true',
        help='record an allocator history and dump it; adds overhead, so run '
             'this as a separate diagnostic pass, never with timing results',
    )
    parser.add_argument('--snapshot-out')
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
    if args.mlp_chunk_rows <= 0 or args.mlp_chunk_rows % 256:
        parser.error('--mlp-chunk-rows must be a positive multiple of 256')
    if args.warmup < 0 or args.forwards <= 0:
        parser.error('iteration arguments are invalid')
    if args.snapshot_out and not args.record_history:
        parser.error('--snapshot-out requires --record-history')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def main(argv=None):
    args = parse_args(argv)

    import torch

    import h3_optimizations.memory.linear as memory_linear
    from h3_optimizations.attention.sparse.config import (
        HybridSparseConfig,
        MODE_SAGE128_FUSED_QKV,
    )
    from h3_optimizations.attention.sparse.backend import HybridSparseBackend
    from h3_optimizations.attention.sparse.sparse_sage import (
        load_sparse_sage_spec,
    )
    from h3_optimizations.attention_forward import make_forward as make_attention
    from h3_optimizations.memory.config import (
        MODE_CONVROT_2SLICE,
        ActivationMemoryConfig,
    )
    from h3_optimizations.memory.forward import make_forward as make_block_forward
    from h3_optimizations.qkv.formats import describe_linear
    from h3_optimizations.qkv.projectors import SparseFusedQKVProjector
    from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    allocator_backend = torch.cuda.get_allocator_backend()
    if allocator_backend != 'native':
        raise SystemExit(
            'this experiment measures the native caching allocator; '
            'the active backend is %r' % allocator_backend
        )

    checkpoint = resolve_checkpoint(args.checkpoint)
    block, prefix, hidden, t_dim = build_block(
        torch,
        checkpoint,
        args.block,
        device,
    )

    qkv_format = describe_linear(block.attn.qkv_proj)
    if not qkv_format.convrot_int8_256:
        raise SystemExit(
            'chunked Sparse Sage QKV requires ConvRot-256 TensorWise INT8; '
            'checkpoint QKV is %s' % qkv_format.label
        )

    # Read the logical widths from the Linear modules, not from the weights:
    # a quantized weight's stored shape depends on its layout.
    qkv_width = 3 * int(block.attn.heads) * int(block.attn.head_dim)
    fc1_width = int(block.mlp.fc1.out_features)
    if qkv_width != int(block.attn.qkv_proj.out_features):
        raise SystemExit('H3 attention head geometry disagrees with qkv_proj')
    if fc1_width % 4:
        raise SystemExit('H3 fc1 width must split into two SwiGLU tiles')
    # Each ConvRot tile carries gate and up features for half the SwiGLU
    # width, so one tile expansion is [rows, fc1_width / 2] BF16.
    mlp_tile_width = fc1_width // 2
    item_size = torch.finfo(torch.bfloat16).bits // 8

    prediction = predict_reuse(
        args.qkv_chunk_rows,
        args.mlp_chunk_rows,
        qkv_width,
        mlp_tile_width,
        item_size,
    )

    spec = load_sparse_sage_spec()
    projector = SparseFusedQKVProjector(
        spec,
        required=True,
        chunk_rows=args.qkv_chunk_rows,
    )
    backend = HybridSparseBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128_FUSED_QKV,
            video_budget=args.video_budget,
        ),
        kernel_spec=spec,
        projector=projector,
    )
    block.attn.forward = make_attention(
        block.attn,
        args.block,
        backend=backend,
        projector=projector,
    )
    memory_config = ActivationMemoryConfig(
        mode=MODE_CONVROT_2SLICE,
        chunk_rows=args.mlp_chunk_rows,
        strict=True,
    )
    block_forward = make_block_forward(
        block,
        args.block,
        memory_config,
        original_forward=block.forward,
    )

    layout = make_layout(args.sequence, args.video_start)
    mod_segments = mod_segments_from_layout(layout)
    options = {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=0,
            step_index=0,
            total_steps=max(1, int(args.forwards)),
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
    # The block forward mutates x in place and returns it. Restore from a
    # persistent copy instead of cloning, so the measured series never
    # allocates a sequence-sized tensor of its own.
    x = reference.clone()
    t_emb = torch.randn(
        (1, t_dim),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)

    trace = AllocationTrace(qkv_width, mlp_tile_width)
    uninstall = install_tracing(block, trace, memory_linear)

    def one_forward():
        x.copy_(reference)
        trace.reset()
        return block_forward(
            x,
            t_emb,
            mod_segments,
            rope,
            transformer_options=options,
        )

    try:
        for _ in range(args.warmup):
            result = one_forward()
            torch.cuda.synchronize(device)
            del result

        # One release after warmup gives every arm the same starting cache.
        # Nothing below this line may clear the cache again.
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)
        stats_before = memory_stats(torch, device)

        if args.record_history:
            torch.cuda.memory._record_memory_history(max_entries=250_000)

        trace.enabled = True
        series = []
        traces = []
        stats_after_first = None
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for index in range(args.forwards):
            start.record()
            result = one_forward()
            stop.record()
            stop.synchronize()
            del result
            elapsed = float(start.elapsed_time(stop))
            series.append(series_point(torch, device, elapsed))
            traces.append(
                summarize_trace(
                    trace,
                    prediction['qkv_request_bytes'],
                    prediction['mlp_request_bytes'],
                )
            )
            if index == 0:
                stats_after_first = memory_stats(torch, device)
                first_addresses = {
                    'qkv': address_histogram(trace.qkv),
                    'fc1': address_histogram(trace.fc1),
                }

        stats_after_last = memory_stats(torch, device)
        last_addresses = {
            'qkv': address_histogram(trace.qkv),
            'fc1': address_histogram(trace.fc1),
        }
        histogram = segment_histogram(torch)
        under_test = segments_of_size(
            torch,
            (
                prediction['qkv_segment_bytes'],
                prediction['mlp_segment_bytes'],
            ),
        )

        if args.record_history:
            snapshot_path = Path(
                args.snapshot_out
                or (
                    'h3_allocator_qkv%d_mlp%d.pickle'
                    % (args.qkv_chunk_rows, args.mlp_chunk_rows)
                )
            ).resolve()
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)
        else:
            snapshot_path = None
    finally:
        trace.enabled = False
        uninstall()

    steady = series[len(series) // 2:]
    reuse_flags = [
        entry['first_full_fc1_reuses_qkv_address'] for entry in traces
    ]
    result = {
        'label': args.label or (
            'qkv%d_mlp%d' % (args.qkv_chunk_rows, args.mlp_chunk_rows)
        ),
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': int(args.block),
        'sequence': int(args.sequence),
        'video_start': int(args.video_start),
        'hidden': int(hidden),
        'qkv_width': qkv_width,
        'mlp_tile_width': mlp_tile_width,
        'qkv_chunk_rows': int(args.qkv_chunk_rows),
        'mlp_chunk_rows': int(args.mlp_chunk_rows),
        'mod_segments': [list(segment) for segment in mod_segments],
        'allocator': {
            'backend': allocator_backend,
            'PYTORCH_CUDA_ALLOC_CONF': os.environ.get(
                'PYTORCH_CUDA_ALLOC_CONF'
            ),
            'matches_production_conf': (
                os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
                == PRODUCTION_ALLOC_CONF
            ),
        },
        'prediction': prediction,
        'reuse': {
            'first_full_fc1_reuses_qkv_address': reuse_flags,
            'reused_every_forward': all(flag is True for flag in reuse_flags),
            'reused_any_forward': any(flag is True for flag in reuse_flags),
        },
        'timing': {
            'median_ms': statistics.median(
                point['elapsed_ms'] for point in series
            ),
            'steady_median_ms': statistics.median(
                point['elapsed_ms'] for point in steady
            ),
            'min_ms': min(point['elapsed_ms'] for point in series),
        },
        'stats': {
            'before_series': stats_before,
            'after_first_forward': stats_after_first,
            'after_last_forward': stats_after_last,
        },
        'series': series,
        'traces': traces,
        'addresses': {
            'after_first_forward': first_addresses,
            'after_last_forward': last_addresses,
        },
        'segments': {
            'histogram': histogram,
            'under_test': under_test,
        },
        'sparse_contract': {
            'architecture': spec.architecture,
            'q_tile': spec.q_tile,
            'kv_tile': spec.kv_tile,
            'v_format': spec.v_format,
        },
        'gpu': {
            'name': torch.cuda.get_device_name(device),
            'capability': list(torch.cuda.get_device_capability(device)),
            'total_memory_bytes': int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        },
        'torch_version': torch.__version__,
        'memory_history_snapshot': (
            None if snapshot_path is None else str(snapshot_path)
        ),
    }

    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output).resolve()
        if not output.parent.is_dir():
            raise SystemExit(
                'output directory does not exist: %s' % output.parent
            )
        output.write_text(serialized + '\n', encoding='utf-8')
    if args.json:
        print(serialized)
    else:
        print('arm %s' % result['label'])
        print(
            '  predicted: %s (QKV segment %.1f MiB, MLP request %.1f MiB)'
            % (
                prediction['verdict'],
                prediction['qkv_segment_bytes'] / 2**20,
                prediction['mlp_request_bytes'] / 2**20,
            )
        )
        print(
            '  measured reuse: %s (%d/%d forwards)'
            % (
                result['reuse']['reused_every_forward'],
                sum(1 for flag in reuse_flags if flag is True),
                len(reuse_flags),
            )
        )
        print(
            '  reserved %.3f GiB, inactive split %.1f MiB, large segments %d'
            % (
                stats_after_last.get('reserved_bytes.all.current', 0) / 2**30,
                stats_after_last.get('inactive_split_bytes.all.current', 0)
                / 2**20,
                stats_after_last.get('segment.large_pool.current', 0),
            )
        )
        print('  steady median %.3f ms' % result['timing']['steady_median_ms'])
        if args.output:
            print('  wrote %s' % Path(args.output).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
