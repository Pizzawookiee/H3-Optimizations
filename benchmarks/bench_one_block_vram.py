'''Measure one real H3 block across memory/attention consumers via ComfyUI.

This benchmark targets the peak activation footprint rather than generation
speed.  It builds a 1376x768, 243-frame (10.13 second at 24 fps) H3 request,
sets AIMDO residency to zero, executes layer 0 of the first denoising step, and
then stops through MiniMaxH3VRAMBlockStopZi.  The stop node synchronizes the
block, writes only small route/layout metadata, and never copies or retains an
activation tensor.

The script drives an already-running ComfyUI server and never imports torch.
All GPU work belongs to the ComfyUI server.
'''

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path


# Keep accidental optional imports in this API client from claiming a CUDA
# context.  This does not affect the already-running ComfyUI server.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

from bench_attention_arms import (  # noqa: E402
    BenchError,
    DEFAULT_CLIP,
    DEFAULT_PROMPT,
    DEFAULT_SERVER,
    DEFAULT_VAE,
    FPS,
    ONE_MP,
    Schemas,
    VramSampler,
    gpu_now,
)


DEFAULT_UNET = 'minimax_h3_fl2va_pruned_bf16.safetensors'
DEFAULT_FRAMES = 243

MEMORY_PATCH = (
    'H3MemoryOptimization',
    {
        'precision_mode': 'Preserve native',
        'qkv_streaming_mode': 'Auto',
        'mlp_memory': 'auto',
        'chunk_rows': 4096,
    },
)

SPARSE_COMMON = {
    'video_budget': 0.3,
    'early_steps': 0,
    'early_kv': 0.3,
    'late_steps': 0,
    'late_kv': 0.3,
}


def sparse_patch(backend):
    return (
        'H3SparseAttentionAdvanced',
        {**SPARSE_COMMON, 'backend': backend},
    )


# before_memory consumers install an optimized_attention override which the
# memory node must inspect.  after_memory consumers own nested H3 attention.
# The diagnostic stop is terminal so production block owners never encounter
# its wrapper while composing their own patches.
ARMS = {
    'default_comfy': {
        'label': 'Default Comfy attention + H3 Memory',
        'before_memory': [
            ('ModelAttentionBackend', {'attention': 'pytorch attention'}),
        ],
    },
    'comfy_kitchen': {
        'label': 'Comfy Kitchen attention + H3 Memory',
        'before_memory': [
            ('ModelAttentionBackend', {'attention': 'comfy kitchen attention'}),
        ],
    },
    'sage_builtin': {
        'label': 'ComfyUI SageAttention + H3 Memory',
        'requires_command_sage': True,
    },
    'kj_sage': {
        'label': 'KJ generic SageAttention + H3 Memory',
        'before_memory': [
            ('PathchSageAttentionKJ', {
                'sage_attention': 'auto',
                'allow_compile': False,
            }),
        ],
    },
    'plague_sla': {
        'label': 'PlagueKind SLA 90% sparse + H3 Memory',
        'before_memory': [
            ('H3SLAAttention', {
                'sparsity_ratio': 0.9,
                'block_size': '64',
                'min_seq_len': 8192,
                'dense_last_steps': 0,
                'protect_audio': True,
                'enabled': True,
            }),
        ],
    },
    'sparse_kitchen': {
        'label': 'H3 Sparse Kitchen INT8 (30% video KV)',
        'after_memory': [sparse_patch('Kitchen INT8')],
    },
    'sparse_sage': {
        'label': 'H3 Sparse Sage (30% video KV)',
        'after_memory': [sparse_patch('Sparse Sage')],
    },
    'frost_bf16': {
        'label': 'H3 FROST BF16 SM89 (30% video KV)',
        'after_memory': [sparse_patch('FROST BF16 (SM89)')],
    },
    'bf16_triton': {
        'label': 'H3 BF16 Triton (30% video KV)',
        'after_memory': [sparse_patch('BF16 Triton')],
    },
}

DEFAULT_ARMS = ','.join(ARMS)


async def add_node(graph, schemas, node_id, node_type, overrides=None, links=None):
    graph[node_id] = {
        'class_type': node_type,
        'inputs': await schemas.inputs(node_type, overrides or {}, links or {}),
        '_meta': {'title': node_id},
    }
    return [node_id, 0]


async def add_conditioning(graph, schemas, args):
    await add_node(
        graph, schemas, 'clip', 'CLIPLoader',
        {'clip_name': args.clip, 'type': 'minimax'},
    )
    await add_node(graph, schemas, 'vae', 'VAELoader', {'vae_name': args.vae})
    await add_node(
        graph, schemas, 'cond', 'MiniMaxH3ImageToVideo',
        {
            'prompt': args.prompt,
            'width': args.width,
            'height': args.height,
            'length': args.frames,
        },
        {'clip': ['clip', 0], 'vae': ['vae', 0]},
    )


async def build_prime_prompt(schemas, args):
    graph = {}
    await add_conditioning(graph, schemas, args)
    await add_node(graph, schemas, 'sink', 'PreviewAny', {}, {'source': ['cond', 1]})
    return graph


async def build_free_barrier_prompt(schemas):
    graph = {}
    await add_node(
        graph, schemas, 'free_barrier', 'PreviewAny', {},
        {'source': 'h3-vram-free-barrier-' + uuid.uuid4().hex},
    )
    return graph


async def build_arm_prompt(schemas, arm_name, args):
    graph = {}
    arm = ARMS[arm_name]
    await add_node(graph, schemas, 'loader', 'UNETLoader', {'unet_name': args.unet})
    await add_conditioning(graph, schemas, args)

    model_ref = ['loader', 0]
    index = 0

    async def patch(node_type, overrides):
        nonlocal index, model_ref
        node_id = 'patch%02d_%s' % (index, node_type)
        index += 1
        model_ref = await add_node(
            graph, schemas, node_id, node_type, overrides, {'model': model_ref},
        )

    for node_type, overrides in arm.get('before_memory', ()):
        await patch(node_type, overrides)

    await patch(*MEMORY_PATCH)

    for node_type, overrides in arm.get('after_memory', ()):
        await patch(node_type, overrides)

    await patch('H3AIMDOResidencyLimiter', {'residency': '0 blocks'})
    await patch(
        'MiniMaxH3VRAMBlockStopZi',
        {
            'enabled': True,
            'run_tag': args.run_tag,
            'arm_name': arm_name,
            'layer': 0,
            'step': 0,
            'branch': 'conditional',
            'overwrite': False,
            'notes': (
                'BF16 one-block VRAM benchmark; %dx%d, %d frames, AIMDO 0 blocks'
                % (args.width, args.height, args.frames)
            ),
        },
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
            'latent_image': ['cond', 1],
        },
    )
    await add_node(graph, schemas, 'sink', 'PreviewAny', {}, {'source': ['sample', 0]})
    return graph


async def queue_prompt(session, server, client_id, graph):
    response = await session.post(
        '%s/prompt' % server,
        json={'prompt': graph, 'client_id': client_id},
    )
    body = await response.json()
    if response.status != 200:
        raise BenchError(
            'prompt rejected (%d):\n%s'
            % (response.status, json.dumps(body, indent=2))
        )
    return body['prompt_id']


def is_expected_block_stop(data, expect_stop):
    exception_name = str(data.get('exception_type') or '').rsplit('.', 1)[-1]
    return (
        expect_stop
        and exception_name == 'BlockLabCaptureComplete'
        and 'diagnostic sampling stopped as requested'
        in str(data.get('exception_message') or '')
    )


async def run_prompt(session, server, client_id, graph, timeout, expect_stop):
    import aiohttp

    ws_url = server.replace('https://', 'wss://').replace('http://', 'ws://')
    async with session.ws_connect('%s/ws?clientId=%s' % (ws_url, client_id)) as ws:
        prompt_id = await queue_prompt(session, server, client_id, graph)
        cached_nodes = []
        started = time.perf_counter()
        deadline = started + timeout

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BenchError('timed out waiting for prompt %s' % prompt_id)
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                raise BenchError('timed out waiting for prompt %s' % prompt_id)
            if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise BenchError('websocket closed before prompt %s finished' % prompt_id)
            if message.type is not aiohttp.WSMsgType.TEXT:
                continue

            payload = json.loads(message.data)
            kind = payload.get('type')
            data = payload.get('data') or {}
            if data.get('prompt_id') not in (None, prompt_id):
                continue

            if kind == 'execution_cached':
                cached_nodes.extend(str(value) for value in data.get('nodes') or ())
            elif kind == 'execution_error':
                if not is_expected_block_stop(data, expect_stop):
                    raise BenchError(
                        'execution failed in node %s (%s), %s: %s'
                        % (
                            data.get('node_id'), data.get('node_type'),
                            data.get('exception_type'), data.get('exception_message'),
                        )
                    )
                return {
                    'prompt_id': prompt_id,
                    'wall_seconds': time.perf_counter() - started,
                    'cached_nodes': cached_nodes,
                    'expected_stop': True,
                    'exception': {
                        'type': data.get('exception_type'),
                        'message': data.get('exception_message'),
                        'node_id': data.get('node_id'),
                        'node_type': data.get('node_type'),
                    },
                }
            elif kind == 'execution_interrupted':
                raise BenchError('execution interrupted')
            elif kind == 'execution_success':
                if expect_stop:
                    raise BenchError('prompt completed without reaching the block-lab stop')
                return {
                    'prompt_id': prompt_id,
                    'wall_seconds': time.perf_counter() - started,
                    'cached_nodes': cached_nodes,
                    'expected_stop': False,
                }
            elif kind == 'executing' and data.get('node') is None:
                if expect_stop:
                    continue
                return {
                    'prompt_id': prompt_id,
                    'wall_seconds': time.perf_counter() - started,
                    'cached_nodes': cached_nodes,
                    'expected_stop': False,
                }


async def require_queue_idle(session, server):
    async with session.get('%s/api/queue' % server) as response:
        if response.status != 200:
            raise BenchError('/api/queue returned %d' % response.status)
        queue = await response.json()
    if queue.get('queue_running') or queue.get('queue_pending'):
        raise BenchError('ComfyUI queue is not idle')


async def unload_models(session, server, *, free_memory=False):
    async with session.post(
        '%s/free' % server,
        json={'unload_models': True, 'free_memory': bool(free_memory)},
    ) as response:
        if response.status != 200:
            raise BenchError('/free returned %d' % response.status)
        await response.read()


async def apply_free_request(
    session, server, client_id, schemas, timeout, settle_seconds, *, free_memory,
):
    """Make asynchronous /free flags observable before the next real prompt."""
    await unload_models(session, server, free_memory=free_memory)
    barrier = await build_free_barrier_prompt(schemas)
    await run_prompt(session, server, client_id, barrier, timeout, False)
    await asyncio.sleep(settle_seconds)
    await require_queue_idle(session, server)


async def server_context(session, server):
    async with session.get('%s/api/system_stats' % server) as response:
        if response.status != 200:
            raise BenchError('/api/system_stats returned %d' % response.status)
        payload = await response.json()
    argv = (payload.get('system') or {}).get('argv') or []
    output = None
    for index, value in enumerate(argv[:-1]):
        if value == '--output-directory':
            output = argv[index + 1]
    return payload, Path(output) if output else Path.cwd() / 'output'


def read_report(report_path):
    path = Path(report_path)
    if not path.is_file():
        raise BenchError('VRAM stop report was not written: %s' % path)
    return json.loads(path.read_text(encoding='utf-8'))


def write_results(path, args, context, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'benchmark': 'H3 BF16 one-block VRAM consumer matrix',
        'config': {
            'unet': args.unet,
            'clip': args.clip,
            'vae': args.vae,
            'width': args.width,
            'height': args.height,
            'frames': args.frames,
            'seconds_at_24fps': args.frames / FPS,
            'schedule_steps': args.schedule_steps,
            'sampled_steps': 1,
            'captured_layer': 0,
            'captured_step': 0,
            'aimdo_residency': '0 blocks',
            'memory_precision': 'Preserve native',
            'memory_qkv_streaming': 'Auto',
            'hard_reset_each_arm': True,
            'conditioning_primed_each_arm': bool(args.prime),
            'sparse_video_budget': SPARSE_COMMON['video_budget'],
            'seed': args.seed,
            'run_tag': args.run_tag,
        },
        'server': context,
        'records': records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def render_table(records):
    lines = [
        '| Arm | Peak total GiB | Increase GiB | Block output | Route |',
        '| --- | ---: | ---: | --- | --- |',
    ]
    for row in records:
        if row.get('error'):
            lines.append('| %s | ERROR | - | - | %s |' % (
                row['label'], str(row['error']).replace('|', '\\|'),
            ))
            continue
        route = json.dumps(row.get('routes') or {}, sort_keys=True)
        if len(route) > 120:
            route = route[:117] + '...'
        lines.append(
            '| %s | %.2f | %.2f | %s | `%s` |'
            % (
                row['label'], row['peak_mib'] / 1024.0,
                row['peak_over_baseline_mib'] / 1024.0,
                json.dumps((row.get('measurement') or {}).get('output') or {}),
                route.replace('|', '\\|'),
            )
        )
    return '\n'.join(lines)


async def run_matrix(args):
    import aiohttp

    client_id = str(uuid.uuid4())
    records = []
    async with aiohttp.ClientSession() as session:
        await require_queue_idle(session, args.server)
        stats, output_root = await server_context(session, args.server)
        argv = (stats.get('system') or {}).get('argv') or []
        if any(ARMS[name].get('requires_command_sage') for name in args.arm_list):
            if '--use-sage-attention' not in argv:
                raise BenchError(
                    'sage_builtin requires a server started with --use-sage-attention'
                )

        schemas = Schemas(args.server, session)
        if args.dry_run:
            graph = await build_arm_prompt(
                schemas, args.arm_list[0], args,
            )
            print(json.dumps(graph, indent=2))
            return records, stats

        for arm_name in args.arm_list:
            arm = ARMS[arm_name]
            print('=== %s ===' % arm['label'], flush=True)
            await apply_free_request(
                session, args.server, client_id, schemas, args.timeout,
                args.settle_seconds, free_memory=True,
            )
            if args.prime:
                print('  conditioning prime (discarded)', flush=True)
                prime_graph = await build_prime_prompt(schemas, args)
                await run_prompt(
                    session, args.server, client_id, prime_graph,
                    args.timeout, False,
                )
                await asyncio.sleep(args.settle_seconds)
                await require_queue_idle(session, args.server)
            baseline = gpu_now()
            report_path = output_root / 'h3_vram' / args.run_tag / (arm_name + '.json')
            if report_path.exists():
                raise BenchError(
                    'refusing to overwrite existing benchmark report %s'
                    % report_path
                )
            graph = await build_arm_prompt(schemas, arm_name, args)
            record = {
                'arm': arm_name,
                'label': arm['label'],
                'run_tag': args.run_tag,
                'report_path': str(report_path),
                'baseline_gpu': baseline,
                'peak_mib': None,
                'peak_over_baseline_mib': None,
            }
            sampler = VramSampler(interval_ms=args.sample_ms)
            sampler.start()
            try:
                result = await run_prompt(
                    session, args.server, client_id, graph, args.timeout, True,
                )
                if 'sample' in result['cached_nodes']:
                    raise BenchError('sampler was served from the execution cache')
                record.update(result)
                report = read_report(report_path)
                record['layout'] = report.get('layout')
                record['routes'] = report.get('routes')
                record['sigma'] = report.get('sigma')
                record['measurement'] = report.get('measurement')
            except BenchError as error:
                record['error'] = str(error)
                print('  FAILED: %s' % error, flush=True)
            finally:
                sampler.stop()

            record['peak_mib'] = sampler.peak_mib()
            record['peak_watts'] = sampler.peak_watts()
            if record['peak_mib'] is not None:
                record['peak_over_baseline_mib'] = (
                    record['peak_mib'] - baseline['memory_used_mib']
                )
            records.append(record)
            write_results(args.output_path, args, stats, records)
            if not record.get('error'):
                print(
                    '  peak %.0f MiB (+%.0f MiB), %.2f s, report %s'
                    % (
                        record['peak_mib'], record['peak_over_baseline_mib'],
                        record['wall_seconds'], report_path,
                    ),
                    flush=True,
                )

        await apply_free_request(
            session, args.server, client_id, schemas, args.timeout,
            args.settle_seconds, free_memory=True,
        )
        return records, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Measure real H3 one-block VRAM across attention consumers.'
    )
    parser.add_argument('--server', default=DEFAULT_SERVER)
    parser.add_argument('--arms', default=DEFAULT_ARMS)
    parser.add_argument('--width', type=int, default=ONE_MP[0])
    parser.add_argument('--height', type=int, default=ONE_MP[1])
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES)
    parser.add_argument('--schedule-steps', type=int, default=20)
    parser.add_argument('--sampler', default='res_multistep')
    parser.add_argument('--scheduler', default='simple')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--unet', default=DEFAULT_UNET)
    parser.add_argument('--clip', default=DEFAULT_CLIP)
    parser.add_argument('--vae', default=DEFAULT_VAE)
    parser.add_argument('--timeout', type=float, default=1800.0)
    parser.add_argument('--settle-seconds', type=float, default=6.0)
    parser.add_argument('--sample-ms', type=int, default=50)
    parser.add_argument('--idle-watts', type=float, default=60.0)
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--no-prime', dest='prime', action='store_false')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)

    args.arm_list = [value.strip() for value in args.arms.split(',') if value.strip()]
    unknown = [name for name in args.arm_list if name not in ARMS]
    if unknown:
        parser.error('unknown arm(s): %s' % ', '.join(unknown))
    if not args.arm_list:
        parser.error('no arms selected')
    if args.frames < 1 or args.width < 1 or args.height < 1:
        parser.error('width, height, and frames must be positive')
    if args.sample_ms < 20:
        parser.error('--sample-ms must be at least 20')
    if not args.dry_run and not args.i_understand_this_uses_gpu:
        parser.error(
            'this runs a real model on the GPU; pass '
            '--i-understand-this-uses-gpu after confirming the card is idle'
        )
    if not args.run_tag:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        args.run_tag = 'h3_vram_%s_%s' % (stamp, uuid.uuid4().hex[:8])
    if not args.output:
        args.output = str(Path('.agent') / 'tmp' / (args.run_tag + '.json'))
    args.output_path = Path(args.output).resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        idle = gpu_now()
        print(
            'GPU preflight: %.0f MiB used, %.1f W, %.0f MHz'
            % (idle['memory_used_mib'], idle['power_w'], idle['clocks_sm_mhz'])
        )
        if idle['power_w'] > args.idle_watts:
            raise SystemExit(
                'GPU power %.1f W exceeds the %.1f W idle limit'
                % (idle['power_w'], args.idle_watts)
            )

    try:
        records, context = asyncio.run(run_matrix(args))
    except BenchError as error:
        raise SystemExit(str(error))
    if args.dry_run:
        return 0

    write_results(args.output_path, args, context, records)
    print('\n' + render_table(records))
    print('\nwrote %s' % args.output_path)
    return 0 if records and all(not row.get('error') for row in records) else 1


if __name__ == '__main__':
    sys.exit(main())
