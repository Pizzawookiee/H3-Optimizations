'''Profile one real H3 model pass through all 50 blocks.

This is intentionally separate from ``bench_one_block_vram.py``. It stops after
layer 49 of the first selected conditional pass and compares the full production
boundaries used by the README attention arms. Optional allocator history
recording requires a server started with ``--disable-cuda-malloc``.
'''

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

from bench_attention_arms import (  # noqa: E402
    DEFAULT_CONDITIONING_CACHE_ENTRY,
    DEFAULT_CONDITIONING_TOKENS,
    DEFAULT_SERVER,
    ONE_MP,
    Schemas,
    VramSampler,
    gpu_now,
)
from bench_one_block_vram import (  # noqa: E402
    BenchError,
    add_node,
    apply_free_request,
    build_free_barrier_prompt,
    read_report,
    require_queue_idle,
    run_prompt,
    server_context,
)


DEFAULT_UNET = 'hf_minimax_h3\\minimax_h3_fl2va_pruned_int8_convrot.safetensors'
DEFAULT_FRAMES = 124

MEMORY_PATCH = (
    'H3MemoryOptimization',
    {
        'precision_mode': 'Preserve native',
        'qkv_streaming_mode': 'Auto',
        'mlp_memory': 'auto',
        'chunk_rows': 4096,
    },
)

ARMS = {
    'kitchen_baseline': {
        'label': 'Comfy Kitchen INT8 dense',
        'patches': [
            ('ModelAttentionBackend', {'attention': 'comfy kitchen attention'}),
        ],
    },
    'kitchen_memory': {
        'label': 'Comfy Kitchen INT8 dense + H3 Memory',
        'patches': [
            ('ModelAttentionBackend', {'attention': 'comfy kitchen attention'}),
            MEMORY_PATCH,
        ],
    },
    'native_kv100': {
        'label': 'Native INT8 attention (KV 100%) + H3 Memory',
        'patches': [
            MEMORY_PATCH,
            ('H3SparseAttentionAdvanced', {
                'video_budget': 1.0,
                'early_steps': 0,
                'early_kv': 1.0,
                'late_steps': 0,
                'late_kv': 1.0,
                'backend': 'Kitchen INT8',
            }),
        ],
    },
    'native_kv30': {
        'label': 'Native INT8 attention (KV 30%) + H3 Memory',
        'patches': [
            MEMORY_PATCH,
            ('H3SparseAttentionAdvanced', {
                'video_budget': 0.3,
                'early_steps': 0,
                'early_kv': 0.3,
                'late_steps': 0,
                'late_kv': 0.3,
                'backend': 'Kitchen INT8',
            }),
        ],
    },
    'native_kv30_lower_vram': {
        'label': 'Native INT8 attention (KV 30%) + H3 Memory (Lower VRAM)',
        'patches': [
            (
                'H3MemoryOptimization',
                {
                    'precision_mode': 'Preserve native',
                    'qkv_streaming_mode': 'Auto',
                    'mlp_memory': 'auto',
                    'chunk_rows': 4096,
                    'kitchen_v_memory_mode': 'Lower VRAM (slower)',
                },
            ),
            ('H3SparseAttentionAdvanced', {
                'video_budget': 0.3,
                'early_steps': 0,
                'early_kv': 0.3,
                'late_steps': 0,
                'late_kv': 0.3,
                'backend': 'Kitchen INT8',
            }),
        ],
    },
}


async def build_arm_prompt(schemas, arm_name, args):
    graph = {}
    await add_node(graph, schemas, 'loader', 'UNETLoader', {'unet_name': args.unet})
    await add_node(
        graph, schemas, 'cond', 'H3BenchmarkSyntheticConditioning',
        {
            'tokens': args.conditioning_tokens,
            'cache_entry': args.conditioning_cache_entry,
        },
    )
    await add_node(
        graph, schemas, 'latent', 'EmptyMiniMaxH3LatentAV',
        {'width': args.width, 'height': args.height, 'length': args.frames},
    )
    model_ref = ['loader', 0]

    patches = (
        [('H3BenchmarkForceQKVConfig0', {})]
        + ARMS[arm_name]['patches']
    )
    for index, (node_type, overrides) in enumerate(patches):
        node_id = 'patch%02d_%s' % (index, node_type)
        model_ref = await add_node(
            graph, schemas, node_id, node_type, overrides, {'model': model_ref},
        )

    model_ref = await add_node(
        graph, schemas, 'residency', 'H3AIMDOResidencyLimiter',
        {'residency': '0 blocks'}, {'model': model_ref},
    )
    model_ref = await add_node(
        graph, schemas, 'profile', 'MiniMaxH3OneStepProfileZi',
        {
            'enabled': True,
            'run_tag': args.run_tag,
            'arm_name': arm_name,
            'step': 0,
            'branch': 'conditional',
            'allocator_snapshot': args.allocator_snapshot,
            'overwrite': False,
            'notes': 'one conditional pass through all 50 H3 blocks',
        },
        {'model': model_ref},
    )
    await add_node(
        graph, schemas, 'guider', 'BasicGuider', {},
        {'model': model_ref, 'conditioning': ['cond', 0]},
    )
    await add_node(
        graph, schemas, 'sampler', 'KSamplerSelect',
        {'sampler_name': args.sampler},
    )
    await add_node(
        graph, schemas, 'schedule', 'BasicScheduler',
        {'scheduler': args.scheduler, 'steps': args.schedule_steps, 'denoise': 1.0},
        {'model': ['loader', 0]},
    )
    await add_node(
        graph, schemas, 'split', 'SplitSigmas', {'step': 1},
        {'sigmas': ['schedule', 0]},
    )
    await add_node(graph, schemas, 'noise', 'RandomNoise', {'noise_seed': args.seed})
    await add_node(
        graph, schemas, 'sample', 'SamplerCustomAdvanced', {},
        {
            'noise': ['noise', 0],
            'guider': ['guider', 0],
            'sampler': ['sampler', 0],
            'sigmas': ['split', 0],
            'latent_image': ['latent', 0],
        },
    )
    await add_node(graph, schemas, 'sink', 'PreviewAny', {}, {'source': ['sample', 0]})
    return graph


def write_results(path, args, server, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'schema_version': 1,
        'benchmark': 'H3 one-step all-layer profile',
        'config': {
            'arms': args.arm_list,
            'width': args.width,
            'height': args.height,
            'frames': args.frames,
            'schedule_steps': args.schedule_steps,
            'allocator_snapshot': args.allocator_snapshot,
            'conditioning_tokens': args.conditioning_tokens,
            'conditioning_cache_entry': args.conditioning_cache_entry,
            'unet': args.unet,
            'run_tag': args.run_tag,
        },
        'server': server,
        'records': records,
    }, indent=2), encoding='utf-8')


async def run_matrix(args):
    import aiohttp

    client_id = str(uuid.uuid4())
    records = []
    async with aiohttp.ClientSession() as session:
        await require_queue_idle(session, args.server)
        server, output_root = await server_context(session, args.server)
        schemas = Schemas(args.server, session)

        for arm_name in args.arm_list:
            print('=== %s ===' % ARMS[arm_name]['label'], flush=True)
            await apply_free_request(
                session, args.server, client_id, schemas, args.timeout,
                args.settle_seconds, free_memory=True,
            )
            baseline = gpu_now()
            report_path = output_root / 'h3_vram' / args.run_tag / (arm_name + '.json')
            if report_path.exists():
                raise BenchError('refusing to overwrite %s' % report_path)
            graph = await build_arm_prompt(schemas, arm_name, args)
            sampler = VramSampler(interval_ms=args.sample_ms)
            sampler.start()
            try:
                result = await run_prompt(
                    session, args.server, client_id, graph, args.timeout, True,
                    expect_interrupt=True,
                )
                report = read_report(report_path)
            finally:
                sampler.stop()
            record = {
                'arm': arm_name,
                'label': ARMS[arm_name]['label'],
                'wall_seconds': result['wall_seconds'],
                'baseline_gpu': baseline,
                'card_peak_mib': sampler.peak_mib(),
                'card_peak_watts': sampler.peak_watts(),
                'report_path': str(report_path),
                'profile': report,
            }
            records.append(record)
            write_results(args.output_path, args, server, records)
            measurement = report['measurement']
            print(
                '  all layers %.3f ms, allocated peak %.1f MiB, card peak %.0f MiB'
                % (
                    measurement['timing']['all_layers_gpu_ms'],
                    measurement['memory']['peak_allocated_bytes'] / (1024.0 * 1024.0),
                    record['card_peak_mib'],
                ),
                flush=True,
            )

        await apply_free_request(
            session, args.server, client_id, schemas, args.timeout,
            args.settle_seconds, free_memory=True,
        )
        return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Profile one H3 conditional pass through all 50 blocks.'
    )
    parser.add_argument('--server', default=DEFAULT_SERVER)
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--width', type=int, default=ONE_MP[0])
    parser.add_argument('--height', type=int, default=ONE_MP[1])
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES)
    parser.add_argument('--schedule-steps', type=int, default=20)
    parser.add_argument('--sampler', default='res_multistep')
    parser.add_argument('--scheduler', default='simple')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--unet', default=DEFAULT_UNET)
    parser.add_argument('--conditioning-tokens', type=int, default=DEFAULT_CONDITIONING_TOKENS)
    parser.add_argument(
        '--conditioning-cache-entry', default=DEFAULT_CONDITIONING_CACHE_ENTRY,
    )
    parser.add_argument('--timeout', type=float, default=1800.0)
    parser.add_argument('--settle-seconds', type=float, default=6.0)
    parser.add_argument('--sample-ms', type=int, default=50)
    parser.add_argument('--allocator-snapshot', action='store_true')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    args.arm_list = [name.strip() for name in args.arms.split(',') if name.strip()]
    unknown = [name for name in args.arm_list if name not in ARMS]
    if unknown:
        parser.error('unknown arm(s): %s' % ', '.join(unknown))
    if not args.arm_list:
        parser.error('no arms selected')
    if min(args.width, args.height, args.frames, args.schedule_steps) < 1:
        parser.error('dimensions, frames, and schedule steps must be positive')
    if not args.dry_run and not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu after confirming the card is idle')
    if not args.run_tag:
        args.run_tag = 'h3_one_step_%s_%s' % (
            time.strftime('%Y%m%d_%H%M%S'), uuid.uuid4().hex[:8],
        )
    if not args.output:
        args.output = str(Path('.agent') / 'tmp' / (args.run_tag + '.json'))
    args.output_path = Path(args.output).resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        async def show():
            import aiohttp
            async with aiohttp.ClientSession() as session:
                schemas = Schemas(args.server, session)
                print(json.dumps(
                    await build_arm_prompt(schemas, args.arm_list[0], args), indent=2,
                ))
        asyncio.run(show())
        return 0
    idle = gpu_now()
    print('GPU preflight: %.0f MiB, %.1f W, %.0f MHz' % (
        idle['memory_used_mib'], idle['power_w'], idle['clocks_sm_mhz'],
    ))
    asyncio.run(run_matrix(args))
    print('wrote %s' % args.output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
