'''Compare H3 sparse-attention backends on identical Q/K/V and routing.'''

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib.metadata
from itertools import combinations
import json
from pathlib import Path
import statistics
import subprocess
import sys


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = next(
    (parent for parent in PACK_ROOT.parents if (parent / 'comfy').is_dir()),
    PACK_ROOT.parents[1],
)
BENCHMARK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PACK_ROOT))

from bench_chunked_kitchen_qkv import cuda_kernel_names  # noqa: E402
from bench_chunked_sparse_qkv import (  # noqa: E402
    benchmark_round_robin,
    make_layout,
    tensor_metrics,
)


ARM_ORDER = ('int8_triton', 'fp8_flex', 'sparse_sage', 'plaguekind_sla')
DEFAULT_SEQUENCE = 54006
DEFAULT_HEADS = 56
DEFAULT_PARITY_SEQUENCE = 1024
DEFAULT_VIDEO_BUDGET = 0.3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Compare INT8 Triton, FP8 FlexAttention, Sparse Sage, and the '
            'PlagueKind SLA kernel from identical already-projected H3 Q/K/V.'
        )
    )
    parser.add_argument('--sequence', type=int, default=DEFAULT_SEQUENCE)
    parser.add_argument('--heads', type=int, default=DEFAULT_HEADS)
    parser.add_argument('--video-start', type=int, default=256)
    parser.add_argument('--video-budget', type=float, default=DEFAULT_VIDEO_BUDGET)
    parser.add_argument('--parity-sequence', type=int, default=DEFAULT_PARITY_SEQUENCE)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--skip-parity', action='store_true')
    parser.add_argument('--skip-kernel-trace', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if args.sequence <= 0 or args.heads <= 0 or args.parity_sequence <= 0:
        parser.error('sequence, heads, and parity sequence must be positive')
    if not 0 < args.video_start < args.sequence:
        parser.error('--video-start must be inside the sequence')
    if not 0.01 <= args.video_budget <= 1.0:
        parser.error('--video-budget must be in [0.01, 1]')
    if args.device < 0:
        parser.error('--device must be non-negative')
    if args.warmup < 0 or args.iterations <= 0:
        parser.error('iteration arguments are invalid')
    if not args.i_understand_this_uses_gpu:
        parser.error(
            'pass --i-understand-this-uses-gpu after the required idle-GPU preflight'
        )
    return args


def validate_geometry(contracts):
    geometry = {
        name: (int(spec.q_tile), int(spec.kv_tile))
        for name, spec in contracts.items()
    }
    if len(set(geometry.values())) != 1:
        raise ValueError(
            'backend route geometries differ: %s'
            % ', '.join(
                '%s=%dQx%dKV' % (name, *value)
                for name, value in geometry.items()
            )
        )
    return next(iter(geometry.values()))


def absolute_lut(torch, lut):
    return torch.cumsum(lut, dim=-1, dtype=torch.int32)


def runtime_options(sequence, layout, dtype, device):
    from h3_optimizations.runtime.context import RUNTIME_KEY, RuntimeSnapshot

    return {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=0,
            step_index=5,
            total_steps=20,
            layout=layout,
            compute_dtype=dtype,
            device=device,
        )
    }


def prepared_payload(name, prepared):
    if name in ('int8_triton', 'sparse_sage'):
        return prepared.sparse
    return prepared


def prepared_route(torch, name, prepared):
    payload = prepared_payload(name, prepared)
    if name == 'fp8_flex':
        return payload.block_mask.kv_num_blocks, payload.block_mask.kv_indices
    if name in ('int8_triton', 'plaguekind_sla') and not hasattr(payload, 'lut'):
        indices = torch.zeros(
            (*payload.valid_block_num.shape, payload.kv_tiles),
            dtype=torch.int32,
            device=payload.valid_block_num.device,
        )
        if payload.dense_q_tiles:
            dense_indices = torch.arange(
                payload.kv_tiles,
                dtype=torch.int32,
                device=indices.device,
            )
            indices[..., :payload.dense_q_tiles, :] = dense_indices
        if payload.sparse_q_tiles:
            sparse_end = payload.dense_q_tiles + payload.sparse_q_tiles
            indices[
                ...,
                payload.dense_q_tiles:sparse_end,
                :payload.sparse_selected,
            ] = payload.kv_indices
        return payload.valid_block_num, indices
    return payload.valid_block_num, absolute_lut(torch, payload.lut)


def route_comparison(torch, name, prepared, expected_lut, expected_valid):
    valid, indices = prepared_route(torch, name, prepared)
    expected_indices = absolute_lut(torch, expected_lut)
    result = {
        'valid_shape': list(valid.shape),
        'indices_shape': list(indices.shape),
        'valid_exact': bool(torch.equal(valid, expected_valid)),
        'indices_exact': False,
        'different_selected_indices': None,
    }
    if tuple(indices.shape) != tuple(expected_indices.shape):
        return result
    slots = torch.arange(indices.shape[-1], device=indices.device)
    selected = slots < expected_valid[..., None]
    different = (indices != expected_indices) & selected
    result['different_selected_indices'] = int(different.sum().item())
    result['indices_exact'] = result['different_selected_indices'] == 0
    return result


def require_matching_routes(routes):
    mismatches = [
        name
        for name, details in routes.items()
        if not details['valid_exact'] or not details['indices_exact']
    ]
    if mismatches:
        raise RuntimeError(
            'backend sparse routes differ from the shared router: %s'
            % ', '.join(mismatches)
        )


def _tensor_storages(torch, value, storages):
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        key = (str(value.device), int(storage.data_ptr()))
        storages.setdefault(key, int(storage.nbytes()))
        return
    if is_dataclass(value):
        for field in fields(value):
            _tensor_storages(torch, getattr(value, field.name), storages)
        return
    if isinstance(value, dict):
        for item in value.values():
            _tensor_storages(torch, item, storages)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _tensor_storages(torch, item, storages)
        return
    for name in (
        'kv_num_blocks',
        'kv_indices',
        'full_kv_num_blocks',
        'full_kv_indices',
        'q_num_blocks',
        'q_indices',
    ):
        tensor = getattr(value, name, None)
        if torch.is_tensor(tensor):
            _tensor_storages(torch, tensor, storages)


def prepared_tensor_bytes(torch, prepared):
    storages = {}
    _tensor_storages(torch, prepared, storages)
    return sum(storages.values())


def describe_fp8_carrier(prepared, spec):
    if hasattr(prepared, 'q_fp8'):
        return {
            'implementation': 'compiled torch FlexAttention',
            'kernel_backend': spec.kernel_backend,
            'qkv': 'FP8 Q/K/V',
            'qk_scale': 'combined per head FP32',
            'v_scale': 'per head FP32',
        }
    return {
        'implementation': 'compiled torch FlexAttention',
        'kernel_backend': getattr(spec, 'kernel_backend', 'TRITON'),
        'qkv': 'floating Q, FP8 K/V',
        'kv_scale': 'per head FP32',
    }


def complete_attention(arm):
    prepared = arm['prepare']()
    try:
        return arm['backend'].execute(prepared)
    finally:
        del prepared


def benchmark_execution(torch, arms, warmup, iterations, device):
    for _ in range(warmup):
        for name in ARM_ORDER:
            prepared = arms[name]['prepare']()
            output = arms[name]['backend'].execute(prepared)
            torch.cuda.synchronize(device)
            del output, prepared
    torch.cuda.empty_cache()

    samples = {name: [] for name in ARM_ORDER}
    carrier_live = {name: [] for name in ARM_ORDER}
    carrier_tensor = {name: [] for name in ARM_ORDER}
    peak_from_inputs = {name: [] for name in ARM_ORDER}
    execution_peak = {name: [] for name in ARM_ORDER}
    output_live = {name: [] for name in ARM_ORDER}
    for iteration in range(iterations):
        order = ARM_ORDER if iteration % 2 == 0 else tuple(reversed(ARM_ORDER))
        for name in order:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            prepared = arms[name]['prepare']()
            torch.cuda.synchronize(device)
            after_prepare = torch.cuda.memory_allocated(device)
            carrier_live[name].append(int(after_prepare - baseline))
            carrier_tensor[name].append(prepared_tensor_bytes(torch, prepared))

            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            output = arms[name]['backend'].execute(prepared)
            stop.record()
            stop.synchronize()
            samples[name].append(float(start.elapsed_time(stop)))
            peak = torch.cuda.max_memory_allocated(device)
            peak_from_inputs[name].append(int(peak - baseline))
            execution_peak[name].append(int(peak - after_prepare))
            output_live[name].append(
                int(torch.cuda.memory_allocated(device) - after_prepare)
            )
            del output, prepared
            torch.cuda.synchronize(device)
    return {
        name: {
            'median_ms': statistics.median(samples[name]),
            'min_ms': min(samples[name]),
            'samples_ms': samples[name],
            'prepared_live_bytes': max(carrier_live[name]),
            'prepared_tensor_bytes': max(carrier_tensor[name]),
            'peak_from_float_inputs_bytes': max(peak_from_inputs[name]),
            'execution_incremental_peak_bytes': max(execution_peak[name]),
            'output_live_bytes': max(output_live[name]),
        }
        for name in ARM_ORDER
    }


def sparse_float_reference(torch, q, k, v, lut, valid, q_tile, kv_tile):
    absolute = absolute_lut(torch, lut).cpu()
    counts = valid.cpu()
    output = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    sequence = q.shape[-2]
    for head in range(q.shape[1]):
        for q_block in range(absolute.shape[-2]):
            q_start = q_block * q_tile
            q_end = min(q_start + q_tile, sequence)
            count = int(counts[0, head, q_block])
            blocks = absolute[0, head, q_block, :count].tolist()
            positions = []
            for block in blocks:
                start = int(block) * kv_tile
                positions.extend(range(start, min(start + kv_tile, sequence)))
            indices = torch.tensor(positions, dtype=torch.long, device=q.device)
            q_rows = q[0, head, q_start:q_end].float()
            k_rows = k[0, head].index_select(0, indices).float()
            v_rows = v[0, head].index_select(0, indices).float()
            probabilities = torch.softmax(q_rows @ k_rows.T * scale, dim=-1)
            output[0, head, q_start:q_end].copy_(
                (probabilities @ v_rows).to(output.dtype)
            )
    return output


def parity_report(torch, arms, q, k, v, config, q_tile, kv_tile, sequence, video_start):
    from h3_optimizations.attention.sparse.router import SparseTileRouter

    q = q[..., :sequence, :].clone()
    k = k[..., :sequence, :].clone()
    v = v[..., :sequence, :].clone()
    layout = make_layout(sequence, min(video_start, sequence - 1))
    options = runtime_options(sequence, layout, q.dtype, q.device)
    router = SparseTileRouter(config, q_tile=q_tile, kv_tile=kv_tile)
    expected_lut, expected_valid, mask = router.build_lut(
        q, k, layout, config.video_budget
    )
    reference = sparse_float_reference(
        torch, q, k, v, expected_lut, expected_valid, q_tile, kv_tile
    )
    outputs = {}
    routes = {}
    for name in ARM_ORDER:
        backend = arms[name]['backend']
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=0,
            transformer_options=options,
        )
        routes[name] = route_comparison(
            torch, name, prepared, expected_lut, expected_valid
        )
        outputs[name] = backend.execute(prepared)
        del prepared
    require_matching_routes(routes)
    torch.cuda.synchronize(q.device)

    against_reference = {
        name: tensor_metrics(torch, output, reference)
        for name, output in outputs.items()
    }
    pairwise = {
        '%s_vs_%s' % (left, right): tensor_metrics(
            torch, outputs[left], outputs[right]
        )
        for left, right in combinations(ARM_ORDER, 2)
    }
    del outputs, reference, q, k, v, expected_lut, expected_valid
    return {
        'sequence': sequence,
        'route': mask.as_dict(),
        'route_contract': routes,
        'against_float_sparse_reference': against_reference,
        'pairwise': pairwise,
    }


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        module = sys.modules.get(name)
        return getattr(module, '__version__', None)


def git_value(*args):
    completed = subprocess.run(
        ('git', *args),
        cwd=PACK_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def format_bytes(value):
    return '%.3f GiB' % (int(value) / 2**30)


def main(argv=None):
    args = parse_args(argv)

    import torch

    from h3_optimizations.attention.sparse.backend import HybridSparseBackend
    from h3_optimizations.attention.sparse.config import HybridSparseConfig
    from h3_optimizations.attention.sparse.fp8_flex import (
        FP8FlexBackend,
        FP8FlexError,
        preflight_fp8_flex,
    )
    from h3_optimizations.attention.sparse.router import SparseTileRouter
    from h3_optimizations.attention.sparse.sparse_sage import (
        SparseSageError,
        preflight_sparse_sage,
    )
    from h3_optimizations.attention.sparse.triton_sparse import (
        TritonSparseBackend,
        TritonSparseError,
        preflight_triton_sparse,
    )
    from plaguekind_sla import (
        PlagueKindSLABackend,
        PlagueKindSLASpec,
    )

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    cuda_available = lambda: True
    capability_getter = lambda: capability
    try:
        int8_spec = preflight_triton_sparse(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
        )
    except TritonSparseError as exc:
        raise SystemExit('INT8 Triton backend unavailable: %s' % exc) from exc
    try:
        flex_spec = preflight_fp8_flex(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
            device=device,
        )
    except FP8FlexError as exc:
        raise SystemExit('FP8 Flex backend unavailable: %s' % exc) from exc
    try:
        sage_spec = preflight_sparse_sage(
            cuda_available=cuda_available,
            capability_getter=capability_getter,
        )
    except SparseSageError as exc:
        raise SystemExit('Sparse Sage backend unavailable: %s' % exc) from exc
    plaguekind_spec = PlagueKindSLASpec()

    try:
        q_tile, kv_tile = validate_geometry(
            {
                'int8_triton': int8_spec,
                'fp8_flex': flex_spec,
                'sparse_sage': sage_spec,
                'plaguekind_sla': plaguekind_spec,
            }
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dtype = getattr(torch, args.dtype)
    config = HybridSparseConfig(video_budget=args.video_budget)
    layout = make_layout(args.sequence, args.video_start)
    options = runtime_options(args.sequence, layout, dtype, device)
    backends = {
        'int8_triton': TritonSparseBackend(config, spec=int8_spec),
        'fp8_flex': FP8FlexBackend(config, spec=flex_spec),
        'sparse_sage': HybridSparseBackend(config, kernel_spec=sage_spec),
        'plaguekind_sla': PlagueKindSLABackend(config, spec=plaguekind_spec),
    }

    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, args.heads, args.sequence, 128)
    q = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    k = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    v = torch.randn(shape, generator=generator, dtype=dtype, device=device)
    arms = {
        name: {
            'backend': backend,
            'prepare': (
                lambda selected=backend: selected.prepare(
                    q,
                    k,
                    v,
                    layer_index=0,
                    transformer_options=options,
                )
            ),
        }
        for name, backend in backends.items()
    }

    route_router = SparseTileRouter(config, q_tile=q_tile, kv_tile=kv_tile)
    expected_lut, expected_valid, mask = route_router.build_lut(
        q, k, layout, args.video_budget
    )
    route_contract = {}
    prepared_bytes = {}
    fp8_carrier = None
    for name in ARM_ORDER:
        prepared = arms[name]['prepare']()
        route_contract[name] = route_comparison(
            torch, name, prepared, expected_lut, expected_valid
        )
        prepared_bytes[name] = prepared_tensor_bytes(torch, prepared)
        if name == 'fp8_flex':
            fp8_carrier = describe_fp8_carrier(prepared, flex_spec)
        del prepared
    require_matching_routes(route_contract)
    del expected_lut, expected_valid
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    preparation = benchmark_round_robin(
        torch,
        {name: arms[name]['prepare'] for name in ARM_ORDER},
        args.warmup,
        args.iterations,
        device,
    )
    execution = benchmark_execution(
        torch,
        arms,
        args.warmup,
        args.iterations,
        device,
    )
    complete = benchmark_round_robin(
        torch,
        {
            name: (lambda selected=arms[name]: complete_attention(selected))
            for name in ARM_ORDER
        },
        args.warmup,
        args.iterations,
        device,
    )

    parity = None
    if not args.skip_parity:
        parity = parity_report(
            torch,
            arms,
            q,
            k,
            v,
            config,
            q_tile,
            kv_tile,
            min(args.sequence, args.parity_sequence),
            args.video_start,
        )

    kernel_trace = None
    if not args.skip_kernel_trace:
        kernel_trace = {}
        for name in ARM_ORDER:
            prepared = arms[name]['prepare']()
            kernel_trace[name] = cuda_kernel_names(
                torch,
                lambda selected=arms[name], value=prepared: (
                    selected['backend'].execute(value)
                ),
                device,
            )
            del prepared

    result = {
        'boundary': {
            'inputs': (
                'identical synthetic, already-projected HND Q/K/V; H3 QKV '
                'projection and MLP are excluded'
            ),
            'carrier_preparation': 'routing plus each backend carrier conversion',
            'kernel_execution': 'attention kernel from one prebuilt carrier',
            'complete_attention': 'routing, carrier conversion, and attention kernel',
            'memory': (
                'incremental torch.cuda allocated bytes above the persistent '
                'floating Q/K/V inputs; this is not driver-level physical VRAM'
            ),
        },
        'source': {
            'path': str(PACK_ROOT),
            'branch': git_value('branch', '--show-current'),
            'commit': git_value('rev-parse', 'HEAD'),
            'dirty': bool(git_value('status', '--porcelain')),
        },
        'versions': {
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'triton': package_version('triton'),
            'fp8_flex': flex_spec.version,
            'sparse_sage': sage_spec.version,
            'plaguekind_sla_source': plaguekind_spec.source_commit,
        },
        'gpu': {
            'device': args.device,
            'name': torch.cuda.get_device_name(device),
            'capability': list(capability),
        },
        'shape': {
            'batch': 1,
            'heads': args.heads,
            'sequence': args.sequence,
            'head_dim': 128,
            'dtype': str(dtype),
        },
        'routing': {
            'video_start': args.video_start,
            'video_budget': args.video_budget,
            'q_tile': q_tile,
            'kv_tile': kv_tile,
            'metadata': mask.as_dict(),
            'contract': route_contract,
        },
        'backend_contracts': {
            'int8_triton': {
                'implementation': int8_spec.implementation,
                'qkv': 'INT8 Q/K/V',
                'v_scale': 'per KV tile and channel FP32',
            },
            'fp8_flex': fp8_carrier,
            'sparse_sage': {
                'architecture': sage_spec.architecture,
                'kernel': sage_spec.kernel_name,
                'qkv': 'INT8 Q/K, %s V' % sage_spec.v_format.upper(),
                'accumulator': sage_spec.accumulator,
            },
            'plaguekind_sla': {
                'implementation': plaguekind_spec.implementation,
                'source_commit': plaguekind_spec.source_commit,
                'qkv': 'BF16/FP16 contiguous BLHD',
                'route': 'shared route split into dense-prefix and sparse-video launches',
                'accumulator': 'fp32',
            },
        },
        'prepared_tensor_bytes': prepared_bytes,
        'carrier_preparation': preparation,
        'kernel_execution': execution,
        'complete_attention': complete,
        'parity': parity,
        'kernel_trace': kernel_trace,
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
        print(
            'H3 sparse backend comparison: %d heads x %d rows, %.1f%% video KV'
            % (args.heads, args.sequence, args.video_budget * 100.0)
        )
        for boundary, cases in (
            ('carrier preparation', preparation),
            ('kernel execution', execution),
            ('complete attention', complete),
        ):
            print(boundary + ':')
            for name in ARM_ORDER:
                details = cases[name]
                peak = details.get(
                    'peak_from_float_inputs_bytes',
                    details.get('peak_allocated_bytes', 0),
                )
                print(
                    '  %s: %.3f ms, peak %s'
                    % (name, details['median_ms'], format_bytes(peak))
                )
        if parity is not None:
            print('parity against float sparse reference:')
            for name in ARM_ORDER:
                details = parity['against_float_sparse_reference'][name]
                print(
                    '  %s: relative RMSE %.6f, max abs %.6f'
                    % (name, details['relative_rmse'], details['max_abs'])
                )
        if args.output:
            print('wrote %s' % Path(args.output).resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
