'''Measure one real H3 Kitchen QKV chunk-size arm.

The arm uses the shipped Chunked Kitchen QKV -> Sparse Kitchen -> bounded
ConvRot MLP path. Timing and stage-local memory ownership are measured in
separate forwards so the synchronizations required for memory attribution do
not contaminate latency.
'''

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
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

from bench_chunked_kitchen_qkv import make_rope, resolve_checkpoint  # noqa: E402
from bench_h3_block import build_block  # noqa: E402
from bench_h3_block_allocator import (  # noqa: E402
    make_layout,
    memory_stats,
    mod_segments_from_layout,
)


DEFAULT_SEQUENCE = 54_006
DEFAULT_VIDEO_START = 256
DEFAULT_QKV_CHUNK_ROWS = 4_096


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class StageTiming:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.enabled = False
        self.current = None

    def begin_forward(self):
        self.current = {'events': {}, 'counts': {}}

    @contextlib.contextmanager
    def stage(self, name):
        if not self.enabled or self.current is None:
            yield
            return
        start = self.torch.cuda.Event(enable_timing=True)
        stop = self.torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            stop.record()
            self.current['events'].setdefault(str(name), []).append((start, stop))

    def count(self, name):
        if self.enabled and self.current is not None:
            counts = self.current['counts']
            counts[str(name)] = counts.get(str(name), 0) + 1

    def begin_span(self):
        if not self.enabled or self.current is None:
            return None
        start = self.torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def end_span(self, name, start):
        if start is None:
            return
        stop = self.torch.cuda.Event(enable_timing=True)
        stop.record()
        self.current['events'].setdefault(str(name), []).append((start, stop))

    def finish_forward(self):
        result = {}
        for name, events in self.current['events'].items():
            samples = [float(start.elapsed_time(stop)) for start, stop in events]
            result[name] = {
                'gpu_ms': sum(samples),
                'calls': len(samples),
            }
        counts = dict(self.current['counts'])
        self.current = None
        return result, counts


class StageMemory:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device
        self.enabled = False
        self.phases = {}
        self.open_phases = {}

    def _begin(self, name):
        self.torch.cuda.synchronize(self.device)
        baseline = {
            'allocated_bytes': int(self.torch.cuda.memory_allocated(self.device)),
            'reserved_bytes': int(self.torch.cuda.memory_reserved(self.device)),
        }
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self.open_phases[name] = baseline

    def _end(self, name):
        self.torch.cuda.synchronize(self.device)
        baseline = self.open_phases.pop(name)
        peak_allocated = int(self.torch.cuda.max_memory_allocated(self.device))
        peak_reserved = int(self.torch.cuda.max_memory_reserved(self.device))
        self.phases[name] = {
            'baseline_allocated_bytes': baseline['allocated_bytes'],
            'baseline_reserved_bytes': baseline['reserved_bytes'],
            'peak_allocated_bytes': peak_allocated,
            'peak_reserved_bytes': peak_reserved,
            'incremental_allocated_bytes': max(
                0, peak_allocated - baseline['allocated_bytes']
            ),
            'incremental_reserved_bytes': max(
                0, peak_reserved - baseline['reserved_bytes']
            ),
            'end_allocated_bytes': int(
                self.torch.cuda.memory_allocated(self.device)
            ),
            'end_reserved_bytes': int(
                self.torch.cuda.memory_reserved(self.device)
            ),
        }

    def run(self, name, fn):
        if not self.enabled:
            return fn()
        self._begin(name)
        try:
            return fn()
        finally:
            self._end(name)

    def begin_mlp(self):
        if self.enabled and 'mlp' not in self.open_phases:
            self._begin('mlp')

    def finish_mlp(self):
        if 'mlp' in self.open_phases:
            self._end('mlp')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Measure one real Chunked Kitchen QKV -> Sparse Kitchen H3 block arm.'
        )
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--video-start', type=int, default=DEFAULT_VIDEO_START)
    parser.add_argument('--video-budget', type=float, default=0.3)
    parser.add_argument(
        '--qkv-chunk-rows', type=int, default=DEFAULT_QKV_CHUNK_ROWS
    )
    parser.add_argument(
        '--producer-order',
        choices=('full_qkv', 'current', 'route_before_v'),
        default='current',
    )
    parser.add_argument('--diagnostic-stages', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--forwards', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--label', default='')
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
    if args.warmup < 0 or args.forwards <= 0:
        parser.error('iteration arguments are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def summarize_stages(forwards):
    names = sorted({name for forward in forwards for name in forward['stages']})
    result = {}
    for name in names:
        samples = [
            forward['stages'][name]['gpu_ms']
            for forward in forwards
            if name in forward['stages']
        ]
        calls = [
            forward['stages'][name]['calls']
            for forward in forwards
            if name in forward['stages']
        ]
        result[name] = {
            'median_gpu_ms': statistics.median(samples),
            'p10_gpu_ms': percentile(samples, 0.1),
            'p90_gpu_ms': percentile(samples, 0.9),
            'min_gpu_ms': min(samples),
            'median_calls': statistics.median(calls),
        }
    return result


def main(argv=None):
    args = parse_args(argv)

    import torch

    from h3_optimizations.attention.sparse.config import (
        HybridSparseConfig,
        MODE_SAGE128,
        MODE_SAGE128_FUSED_QKV,
    )
    from h3_optimizations.attention.sparse.kitchen_sparse import (
        SparseKitchenError,
        SparseKitchenBackend,
        preflight_sparse_kitchen,
        route_metadata,
        snapshot_for,
    )
    from h3_optimizations.attention.sparse.config import resolve_video_budget
    from h3_optimizations.attention.sparse.router import SparseRouterError
    from h3_optimizations.attention_forward import make_forward as make_attention
    from h3_optimizations.kitchen_qkv import (
        ChunkedKitchenQKVProjector,
        _project_anchor_samples,
        _tile_mean,
        resolve_kitchen,
    )
    from h3_optimizations.memory.config import (
        DEFAULT_CHUNK_ROWS as MLP_CHUNK_ROWS,
        MODE_CONVROT_2SLICE,
        ActivationMemoryConfig,
    )
    from h3_optimizations.memory.forward import make_forward as make_block_forward
    from h3_optimizations.qkv.formats import describe_linear
    from h3_optimizations.qkv.chunked import project_chunk_hnd
    from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot
    from h3_optimizations.mlp_sharing.route import router_kwargs

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda')
    if torch.cuda.get_allocator_backend() != 'native':
        raise SystemExit('the active CUDA allocator is not the production native backend')

    checkpoint = resolve_checkpoint(args.checkpoint)
    block, prefix, hidden, t_dim = build_block(
        torch, checkpoint, args.block, device
    )
    qkv_format = describe_linear(block.attn.qkv_proj)
    if not qkv_format.convrot_int8_256:
        raise SystemExit(
            'Chunked Kitchen QKV requires ConvRot-256 TensorWise INT8; '
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
    producer_spec = producer_kitchen.int8_attention_producer_spec(
        shape, shape, dtype=torch.bfloat16, device=device
    )
    alignment = int(producer_spec.sequence_alignment)
    if args.qkv_chunk_rows % alignment:
        raise SystemExit(
            'QKV chunk rows %d do not satisfy producer alignment %d'
            % (args.qkv_chunk_rows, alignment)
        )

    timing = StageTiming(torch, device)
    memory = StageMemory(torch, device)

    @dataclass
    class PendingV:
        kitchen: object
        producer: object
        retained_v: object
        q_summary: object
        k_summary: object
        timing_start: object = None

    class DeferredVProjector:
        name = 'benchmark_chunked_kitchen_qkv_route_before_v'

        def __init__(self, chunk_rows):
            self.chunk_rows = int(chunk_rows)

        @property
        def installation_signature(self):
            return (self.name, self.chunk_rows)

        def try_project(
            self,
            module,
            x,
            rope_freqs,
            *,
            layer_index,
            transformer_options,
        ):
            del layer_index, transformer_options
            spec = producer_kitchen.int8_attention_producer_spec(
                (1, int(module.heads), int(x.shape[0]), int(module.head_dim)),
                (1, int(module.heads), int(x.shape[0]), int(module.head_dim)),
                dtype=x.dtype,
                device=x.device,
            )
            samples = diagnostic(
                'anchor_projection',
                lambda: _project_anchor_samples(
                    module,
                    x,
                    rope_freqs,
                    spec.k_anchor_positions,
                ),
            )

            def create_producer():
                anchor = producer_kitchen.select_int8_attention_k_anchor(
                    spec, samples
                )
                value = producer_kitchen.create_int8_attention_producer(
                    spec, anchor
                )
                del anchor
                return value

            producer = diagnostic('producer_create', create_producer)
            del samples
            retained_v = None
            q_summaries = []
            k_summaries = []
            for start in range(0, int(x.shape[0]), self.chunk_rows):
                stop = min(start + self.chunk_rows, int(x.shape[0]))
                q, k, v = diagnostic(
                    'chunk_project_norm_rope',
                    lambda start=start, stop=stop: project_chunk_hnd(
                        module, x, rope_freqs, start, stop
                    ),
                )
                if retained_v is None:
                    retained_v = v.new_empty(
                        (1, int(module.heads), int(x.shape[0]), int(module.head_dim))
                    )
                summaries = diagnostic(
                    'routing_summaries',
                    lambda: (
                        _tile_mean(q, int(spec.q_tile)),
                        _tile_mean(k, int(spec.k_tile)),
                    ),
                )
                q_summaries.append(summaries[0])
                k_summaries.append(summaries[1])
                diagnostic(
                    'qk_carrier_pack',
                    lambda: producer_kitchen.quantize_int8_attention_qk_chunk(
                        producer,
                        q,
                        k,
                        q_start=start,
                        k_start=start,
                    ),
                )
                diagnostic(
                    'v_retention',
                    lambda: retained_v[:, :, start:stop, :].copy_(v),
                )
                del q, k, v
            summaries = diagnostic(
                'summary_finalize',
                lambda: (
                    torch.cat(q_summaries, dim=-2),
                    torch.cat(k_summaries, dim=-2),
                ),
            )
            return PendingV(
                kitchen=producer_kitchen,
                producer=producer,
                retained_v=retained_v,
                q_summary=summaries[0],
                k_summary=summaries[1],
            )

    class DeferredVBackend(SparseKitchenBackend):
        def prepare_projected(
            self,
            projected,
            *,
            layer_index,
            transformer_options,
        ):
            if not isinstance(projected, PendingV):
                raise SparseKitchenError('deferred-V benchmark received invalid QKV')
            sequence = int(projected.retained_v.shape[-2])

            def build_route():
                snapshot = snapshot_for(transformer_options, sequence)
                budget = resolve_video_budget(
                    self.config,
                    snapshot.step_index,
                    snapshot.total_steps,
                    layer_index,
                )
                try:
                    return self.router.build_lut_from_summaries(
                        projected.q_summary,
                        projected.k_summary,
                        snapshot.layout,
                        budget,
                        **router_kwargs(transformer_options, layer_index),
                    )
                except SparseRouterError as exc:
                    raise SparseKitchenError(
                        'sparse routing failed: %s' % exc
                    ) from exc

            lut, valid_block_num, mask_metadata = measured(
                'sparse_route_build', build_route
            )

            def finalize_v():
                projected.kitchen.quantize_int8_attention_v(
                    projected.producer, projected.retained_v
                )
                projected.retained_v = None
                return projected.kitchen.finalize_int8_attention_producer(
                    projected.producer
                )

            carrier = measured('qkv_v_finalize', finalize_v)
            timing.end_span('qkv_producer_total', projected.timing_start)

            def prepare_kernel():
                return self.executor.prepare_projected(
                    carrier,
                    lut,
                    valid_block_num,
                    layer_index=layer_index,
                    metadata=route_metadata(
                        mask_metadata,
                        layer_index,
                        projected.q_summary.shape[1],
                    ),
                )

            return measured('sparse_executor_prepare', prepare_kernel)

    qkv_post_starts = []

    class MeasuredFullBackend(SparseKitchenBackend):
        def prepare(
            self,
            q,
            k,
            v,
            *,
            layer_index,
            transformer_options,
        ):
            timing.end_span(
                'qk_norm_rope',
                qkv_post_starts.pop() if qkv_post_starts else None,
            )
            lut, valid_block_num, mask_metadata = measured(
                'sparse_route_build',
                lambda: self._route(
                    q,
                    k,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                ),
            )
            quantized = measured(
                'full_kitchen_pack',
                lambda: self.executor.kitchen.prequantize_int8_attention(
                    q, k, v, cta_k=self.executor.kv_tile
                ),
            )
            return measured(
                'sparse_executor_prepare',
                lambda: self.executor.prepare_projected(
                    quantized,
                    lut,
                    valid_block_num,
                    layer_index=layer_index,
                    metadata=route_metadata(
                        mask_metadata, layer_index, q.shape[1]
                    ),
                ),
            )

    config = HybridSparseConfig(
        mode=(
            MODE_SAGE128
            if args.producer_order == 'full_qkv'
            else MODE_SAGE128_FUSED_QKV
        ),
        video_budget=args.video_budget,
    )
    if args.producer_order == 'full_qkv':
        projector = None
        backend = MeasuredFullBackend(config, kitchen=kitchen)
    elif args.producer_order == 'route_before_v':
        projector = DeferredVProjector(args.qkv_chunk_rows)
        backend = DeferredVBackend(
            config,
            kitchen=kitchen,
            projector=projector,
        )
    else:
        projector = ChunkedKitchenQKVProjector(
            chunk_rows=args.qkv_chunk_rows,
            routing_summaries=True,
        )
        backend = SparseKitchenBackend(
            config,
            kitchen=kitchen,
            projector=projector,
        )

    original_project = None if projector is None else projector.try_project
    original_prepare = backend.prepare_projected
    original_execute = backend.execute
    original_qkv = block.attn.qkv_proj.forward
    original_out = block.attn.out_proj.forward
    pre_kernel_starts = []

    def measured(name, fn):
        with timing.stage(name):
            return memory.run(name, fn)

    def diagnostic(name, fn):
        if not args.diagnostic_stages:
            return fn()
        with timing.stage(name):
            return fn()

    def project(*positional, **keywords):
        if args.producer_order == 'route_before_v':
            producer_start = timing.begin_span()
            result = measured(
                'qkv_qk_producer',
                lambda: original_project(*positional, **keywords),
            )
            result.timing_start = producer_start
        else:
            result = measured(
                'qkv_producer_total',
                lambda: original_project(*positional, **keywords),
            )
        if result is None:
            raise RuntimeError('Chunked Kitchen QKV unexpectedly declined the arm')
        return result

    def prepare(*positional, **keywords):
        if args.producer_order in ('full_qkv', 'route_before_v'):
            return original_prepare(*positional, **keywords)
        return measured(
            'sparse_route_prepare',
            lambda: original_prepare(*positional, **keywords),
        )

    def execute(*positional, **keywords):
        timing.end_span(
            'attention_pre_kernel',
            pre_kernel_starts.pop() if pre_kernel_starts else None,
        )
        return measured(
            'attention_kernel',
            lambda: original_execute(*positional, **keywords),
        )

    def qkv_forward(*positional, **keywords):
        timing.count('qkv_linear')
        if args.producer_order != 'full_qkv' and not args.diagnostic_stages:
            return original_qkv(*positional, **keywords)
        with timing.stage('qkv_linear'):
            result = original_qkv(*positional, **keywords)
        if args.producer_order != 'full_qkv':
            return result
        qkv_post_starts.append(timing.begin_span())
        return result

    def out_forward(*positional, **keywords):
        return measured(
            'attention_out',
            lambda: original_out(*positional, **keywords),
        )

    if projector is not None:
        projector.try_project = project
    backend.prepare_projected = prepare
    backend.execute = execute
    block.attn.qkv_proj.forward = qkv_forward
    block.attn.out_proj.forward = out_forward
    attention_forward = make_attention(
        block.attn,
        args.block,
        backend=backend,
        projector=projector,
    )

    def attention(*positional, **keywords):
        pre_kernel_starts.append(timing.begin_span())
        with timing.stage('attention_total'):
            return attention_forward(*positional, **keywords)

    block.attn.forward = attention
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
        (1, t_dim),
        generator=generator,
        dtype=torch.float32,
        device=device,
    )
    rope = make_rope(torch, args.sequence, device)

    def one_forward():
        x.copy_(reference)
        return block_forward(
            x,
            t_emb,
            mod_segments,
            rope,
            transformer_options=options,
        )

    for _ in range(args.warmup):
        result = one_forward()
        torch.cuda.synchronize(device)
        del result

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    measured_forwards = []
    timing.enabled = True
    for _ in range(args.forwards):
        torch.cuda.reset_peak_memory_stats(device)
        timing.begin_forward()
        with timing.stage('block'):
            result = one_forward()
        torch.cuda.synchronize(device)
        stages, counts = timing.finish_forward()
        measured_forwards.append({
            'stages': stages,
            'counts': counts,
            'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
            'end_allocated_bytes': int(torch.cuda.memory_allocated(device)),
            'end_reserved_bytes': int(torch.cuda.memory_reserved(device)),
        })
        del result
    timing.enabled = False
    stats_after_timing = memory_stats(torch, device)

    torch.cuda.reset_peak_memory_stats(device)
    result = one_forward()
    torch.cuda.synchronize(device)
    uninstrumented_block_peak = {
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
        'peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)),
    }
    del result

    mlp_started = {'value': False}

    def begin_mlp(_module, _inputs):
        if memory.enabled and not mlp_started['value']:
            mlp_started['value'] = True
            memory.begin_mlp()

    hook = block.norm2.register_forward_pre_hook(begin_mlp)
    try:
        memory.enabled = True
        result = one_forward()
        memory.finish_mlp()
        torch.cuda.synchronize(device)
        del result
    finally:
        memory.enabled = False
        hook.remove()

    row_indices = torch.tensor(
        (0, args.sequence // 2, args.sequence - 1),
        dtype=torch.int64,
        device=device,
    )
    column_indices = torch.tensor(
        (0, 1, hidden // 2, hidden - 2, hidden - 1),
        dtype=torch.int64,
        device=device,
    )
    fingerprint = (
        x.index_select(0, row_indices)
        .index_select(1, column_indices)
        .float()
        .cpu()
        .tolist()
    )

    steady = measured_forwards[len(measured_forwards) // 2:]
    result = {
        'label': args.label or 'qkv%d' % args.qkv_chunk_rows,
        'checkpoint': str(checkpoint),
        'checkpoint_prefix': prefix,
        'block': int(args.block),
        'sequence': int(args.sequence),
        'video_start': int(args.video_start),
        'video_budget': float(args.video_budget),
        'seed': int(args.seed),
        'hidden': int(hidden),
        'qkv_chunk_rows': int(args.qkv_chunk_rows),
        'producer_order': args.producer_order,
        'qkv_chunks_per_block': (
            args.sequence + args.qkv_chunk_rows - 1
        ) // args.qkv_chunk_rows,
        'producer_alignment': alignment,
        'mlp_chunk_rows': int(MLP_CHUNK_ROWS),
        'timing': {
            'forwards': measured_forwards,
            'stages': summarize_stages(steady),
        },
        'memory': {
            'uninstrumented_block': uninstrumented_block_peak,
            'phases': memory.phases,
            'stats_after_timing': stats_after_timing,
        },
        'route': backend.as_status(),
        'output_fingerprint': fingerprint,
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
        stages = result['timing']['stages']
        print(
            '%s: pre-kernel %.3f ms, attention %.3f ms, block %.3f ms, '
            'peak %.1f MiB allocated / %.1f MiB reserved'
            % (
                result['label'],
                stages['attention_pre_kernel']['median_gpu_ms'],
                stages['attention_total']['median_gpu_ms'],
                stages['block']['median_gpu_ms'],
                uninstrumented_block_peak['peak_allocated_bytes'] / 2**20,
                uninstrumented_block_peak['peak_reserved_bytes'] / 2**20,
            )
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
