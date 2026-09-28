'''End-to-end comparison of H3 attention and memory configurations.

Unlike the kernel benchmarks in this directory, this drives a running ComfyUI
server with a real H3 checkpoint and sampler. A real cached Qwen conditioning
entry and an empty native H3 latent keep CLIP and VAE out of both timing and
memory measurements without replacing the transformer input with zeros. It
measures user-visible sampler-step cost and the ComfyUI process's dedicated
GPU-memory peak.

Every arm deliberately shares one benchmark control and two residency policies:

* H3BenchmarkForceQKVConfig0 forces compatible ConvRot-256 INT8 QKV linears to
  Comfy Kitchen CUTLASS config 0. This removes the known large-sequence Kitchen
  dispatcher artifact from the comparison.
* speed runs use stock AIMDO residency, matching normal production behavior;
* VRAM runs use ``0 blocks`` so DynamicVRAM cannot hide or amplify activation-
  memory differences by retaining a different amount of H3 weights between
  arms.

The benchmark-only controls are registered by the sibling
``ComfyUI-H3-Extended`` pack. Keep both packs enabled for this script.

The default table is:

1. dense Comfy Kitchen attention;
2. dense SageAttention selected by ComfyUI's ``--use-sage-attention``;
3. dense SageAttention plus H3 Memory Optimization in Preserve native mode.
   ConvRot INT8 QKV is projected in chunks directly into the dense Sage carrier,
   while MLP activations and FinalLayer are chunked;
4. H3 Memory Optimization plus native 64Q x 64KV attention at 100% video KV.
   This shows the additional VRAM effect of streamed BF16 QKV/carrier execution
   without attributing any gain to sparsity;
5. the same native streamed path at a fixed 30% video-KV budget, showing the
   speed and memory effect of sparse attention;
6. fixed 30% video KV with Lower VRAM's two-pass V staging, showing the maximum
   memory saving and its speed cost.

The production ``h3opt`` arm (15% middle budget plus the denser-early ramp) is
still available explicitly. It reallocates roughly the fixed-30% arm's total
attention work toward the quality-sensitive early steps, so a short partial run
would not be a fair timing measurement of that full schedule.

Only a few steps are executed per arm. The schedule is still built at its true
length and then truncated with SplitSigmas, so the executed steps sit on the
real sigma trajectory rather than on a compressed one.

Three numbers per duration, collected by two executions:

  step       median wall time with stock AIMDO residency, warmup discarded
  20-step    projection from that median, matching the README table
  VRAM       first-step peak with AIMDO fixed to zero resident blocks

Dedicated GPU memory comes from the Windows GPU Process Memory counters used by
Task Manager. ``nvidia-smi`` is retained only for the idle/power preflight and a
diagnostic whole-card peak in the JSON record. A discarded priming run loads the
checkpoint before the first measured arm. Nothing between arms resets the
executor.

Run from the ComfyUI root against an already-running server, for example:

    python main.py --use-sage-attention
    .\\.venv\\Scripts\\python.exe custom_nodes\\H3-Optimizations\\benchmarks\\bench_attention_arms.py --i-understand-this-uses-gpu

This script never imports torch and never touches CUDA in-process. All GPU work
happens in the ComfyUI server it talks to.
'''

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_SERVER = 'http://127.0.0.1:8188'
DEFAULT_OUTPUT = r'.agent\tmp\h3-readme-benchmark.json'
DEFAULT_EXPECTED_POWER_LIMIT_W = 200.0
DEFAULT_CONDITIONING_TOKENS = 1450
DEFAULT_CONDITIONING_CACHE_ENTRY = '88b1c73b72bd6b8aaaafc711b27d006a.safetensors'
DEFAULT_ALLOW_ZERO_CONDITIONING = False

DEFAULT_UNET = r'hf_minimax_h3\minimax_h3_fl2va_pruned_int8_convrot.safetensors'
# Shared by the other benchmark clients that still exercise CLIP/VAE paths.
DEFAULT_CLIP = r'hf_minimax_h3\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
DEFAULT_VAE = r'hf_minimax_h3\minimax_h3_video_vae_fp16.safetensors'
DEFAULT_PROMPT = (
    'A slow cinematic dolly shot through a rain-slicked neon alley at night, '
    'steam rising from vents, reflections shimmering on wet asphalt.'
)

ONE_MP = (1376, 768)

FPS = 24
WORKLOADS = {
    '5s': 124,   # 5.17 s
    '10s': 243,  # 10.13 s
}

# Controls applied identically to every arm. They are outside ARMS on purpose:
# selecting an unusual diagnostic arm must not accidentally lose benchmark
# fairness controls.
COMMON_PREFIX = [
    ('H3BenchmarkForceQKVConfig0', {}),
]
COMMON_SUFFIX = [
    ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
]

MEASUREMENT_RESIDENCY = {
    'speed': 'stock',
    'vram': '0 blocks',
}

# Arm-specific patches. The benchmark chain is COMMON_PREFIX + ARMS[arm] +
# COMMON_SUFFIX.
ARMS = {
    'stock': [],
    'pytorch': [
        ('ModelAttentionBackend', {'attention': 'pytorch attention'}),
    ],
    'kitchen': [
        ('ModelAttentionBackend', {'attention': 'comfy kitchen attention'}),
    ],
    'sage': [],
    'sage_memory': [
        ('H3MemoryOptimization', {
            'precision_mode': 'Preserve native',
            'qkv_streaming_mode': 'Auto',
        }),
        ('H3BenchmarkAssertRoute', {
            'attention': 'sage',
            'backend': 'sage_mem_eff',
            'qkv': 'convrot_int8_dense_sage',
            'projector': 'streamed_dense_sage_qkv',
        }),
    ],
    'h3opt_kv100': [
        ('H3MemoryOptimization', {'qkv_streaming_mode': 'Auto'}),
        ('H3SparseAttention', {'video_budget': 1.0, 'denser_early_late_steps': False}),
        ('H3BenchmarkAssertRoute', {
            'attention': 'sparse_kitchen_int8',
            'backend': 'sparse_kitchen_int8',
            'qkv': 'streamed_bf16_kitchen_qkv',
            'projector': 'chunked_kitchen_qkv',
            'video_budget': 1.0,
            'denser_early_late_steps': False,
            'v_memory': 'retain',
        }),
    ],
    'h3opt': [
        ('H3MemoryOptimization', {'qkv_streaming_mode': 'Auto'}),
        ('H3SparseAttention', {'video_budget': 0.15, 'denser_early_late_steps': True}),
        ('H3BenchmarkAssertRoute', {
            'attention': 'sparse_kitchen_int8',
            'backend': 'sparse_kitchen_int8',
            'qkv': 'streamed_bf16_kitchen_qkv',
            'projector': 'chunked_kitchen_qkv',
            'video_budget': 0.15,
            'denser_early_late_steps': True,
            'v_memory': 'retain',
        }),
    ],
    'h3opt_kv30': [
        ('H3MemoryOptimization', {'qkv_streaming_mode': 'Auto'}),
        ('H3SparseAttention', {'video_budget': 0.3, 'denser_early_late_steps': False}),
        ('H3BenchmarkAssertRoute', {
            'attention': 'sparse_kitchen_int8',
            'backend': 'sparse_kitchen_int8',
            'qkv': 'streamed_bf16_kitchen_qkv',
            'projector': 'chunked_kitchen_qkv',
            'video_budget': 0.3,
            'denser_early_late_steps': False,
            'v_memory': 'retain',
        }),
    ],
    'h3opt_kv30_lower_vram': [
        ('H3MemoryOptimization', {
            'qkv_streaming_mode': 'Auto',
            'kitchen_v_memory_mode': 'Lower VRAM (slower)',
        }),
        ('H3SparseAttention', {'video_budget': 0.3, 'denser_early_late_steps': False}),
        ('H3BenchmarkAssertRoute', {
            'attention': 'sparse_kitchen_int8',
            'backend': 'sparse_kitchen_int8',
            'qkv': 'streamed_bf16_kitchen_qkv',
            'projector': 'chunked_kitchen_qkv',
            'video_budget': 0.3,
            'denser_early_late_steps': False,
            'v_memory': 'two_pass',
        }),
    ],
}

DEFAULT_ARMS = 'kitchen,sage,sage_memory,h3opt_kv100,h3opt_kv30,h3opt_kv30_lower_vram'

ARM_LABELS = {
    'stock': 'ComfyUI default attention',
    'pytorch': 'PyTorch attention',
    'kitchen': 'Comfy Kitchen INT8 dense',
    'sage': 'SageAttention dense',
    'sage_memory': 'SageAttention dense + H3 Memory Optimization',
    'h3opt_kv100': 'Native INT8 attention (KV 100%) + H3 Memory Optimization',
    'h3opt': 'Native INT8 attention (15% + early ramp) + H3 Memory Optimization',
    'h3opt_kv30': 'Native INT8 attention (fixed KV 30%) + H3 Memory Optimization',
    'h3opt_kv30_lower_vram': 'Native INT8 attention (fixed KV 30%) + H3 Memory Optimization (Lower VRAM)',
}


# ---------------------------------------------------------------------------
# Server schema
# ---------------------------------------------------------------------------

class Schemas:
    '''Lazily fetched /object_info, used to fill required inputs with defaults.'''

    def __init__(self, server, session):
        self.server = server
        self.session = session
        self.cache = {}

    async def get(self, node_type):
        if node_type not in self.cache:
            url = '%s/object_info/%s' % (self.server, node_type)
            async with self.session.get(url) as response:
                if response.status != 200:
                    suffix = (
                        '; install and enable ComfyUI-H3-Extended'
                        if node_type == 'H3BenchmarkForceQKVConfig0'
                        else ''
                    )
                    raise BenchError(
                        'server has no node %r (status %d)%s'
                        % (node_type, response.status, suffix)
                    )
                payload = await response.json()
            if node_type not in payload:
                raise BenchError('server returned no schema for %r' % node_type)
            self.cache[node_type] = payload[node_type]
        return self.cache[node_type]

    async def inputs(self, node_type, overrides, links):
        '''Every required input: links, then overrides, then schema defaults.'''
        schema = await self.get(node_type)
        required = schema.get('input', {}).get('required', {})
        optional = schema.get('input', {}).get('optional', {})

        unknown = set(overrides) - set(required) - set(optional)
        if unknown:
            raise BenchError(
                'node %s has no input(s) %s; the schema changed'
                % (node_type, ', '.join(sorted(unknown)))
            )

        values = {}
        for name, spec in required.items():
            if name in links:
                values[name] = links[name]
            elif name in overrides:
                values[name] = overrides[name]
            else:
                values[name] = default_for(node_type, name, spec)
        for name in overrides:
            if name in optional:
                values[name] = overrides[name]
        return values


class BenchError(RuntimeError):
    '''A failure that should abort one arm, or the run, with a readable message.'''


def default_for(node_type, name, spec):
    '''The value the UI would show for an input nobody touched.'''
    config = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if 'default' in config:
        return config['default']
    kind = spec[0]
    options = None
    if isinstance(kind, list):
        options = kind
    elif isinstance(config, dict):
        options = config.get('options')
    if options:
        return options[0]
    raise BenchError(
        'node %s input %r is required but has no default; '
        'add an explicit value to its arm definition' % (node_type, name)
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

async def build_prompt(schemas, arm, frames, args, seed=None, measurement='speed'):
    '''The API-format graph for one arm at one duration.'''
    if measurement not in MEASUREMENT_RESIDENCY:
        raise BenchError('unknown measurement %r' % measurement)
    executed = (
        args.warmup_steps + args.measure_steps + 1
        if measurement == 'speed'
        else 1
    )
    if executed >= args.schedule_steps:
        raise BenchError(
            'warmup + measure + 1 (%d) must be under --schedule-steps (%d); '
            'otherwise there is no trajectory left to truncate'
            % (executed, args.schedule_steps)
        )

    graph = {}

    async def node(node_id, node_type, overrides=None, links=None):
        graph[node_id] = {
            'class_type': node_type,
            'inputs': await schemas.inputs(node_type, overrides or {}, links or {}),
            '_meta': {'title': node_id},
        }
        return node_id

    await node('loader', 'UNETLoader', {'unet_name': args.unet})
    await node(
        'cond', 'H3BenchmarkSyntheticConditioning',
        {
            'tokens': args.conditioning_tokens,
            'cache_entry': args.conditioning_cache_entry,
        },
    )
    await node(
        'latent', 'EmptyMiniMaxH3LatentAV',
        {'width': args.width, 'height': args.height, 'length': frames},
    )

    model_ref = ['loader', 0]
    common_suffix = []
    for node_type, overrides in COMMON_SUFFIX:
        overrides = dict(overrides)
        if node_type == 'H3AIMDOResidencyLimiter':
            overrides['residency'] = MEASUREMENT_RESIDENCY[measurement]
        common_suffix.append((node_type, overrides))
    patch_chain = tuple(COMMON_PREFIX) + tuple(ARMS[arm]) + tuple(common_suffix)
    for index, (node_type, overrides) in enumerate(patch_chain):
        patch_id = 'patch%d_%s' % (index, node_type)
        await node(patch_id, node_type, overrides, {'model': model_ref})
        model_ref = [patch_id, 0]

    await node('guider', 'BasicGuider', {},
               {'model': model_ref, 'conditioning': ['cond', 0]})
    await node('sampler', 'KSamplerSelect', {'sampler_name': args.sampler})
    await node(
        'schedule', 'BasicScheduler',
        {'scheduler': args.scheduler, 'steps': args.schedule_steps, 'denoise': 1.0},
        {'model': ['loader', 0]},
    )
    await node('split', 'SplitSigmas', {'step': executed}, {'sigmas': ['schedule', 0]})
    await node(
        'noise', 'RandomNoise',
        {'noise_seed': args.seed if seed is None else seed},
    )
    await node(
        'sample', 'SamplerCustomAdvanced', {},
        {
            'noise': ['noise', 0],
            'guider': ['guider', 0],
            'sampler': ['sampler', 0],
            'sigmas': ['split', 0],
            'latent_image': ['latent', 0],
        },
    )
    await node('sink', 'PreviewAny', {}, {'source': ['sample', 0]})
    return graph, executed


# ---------------------------------------------------------------------------
# VRAM sampling
# ---------------------------------------------------------------------------

class BoardVramSampler:
    '''Diagnostic whole-card VRAM/power from one long-lived nvidia-smi process.'''

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        self.samples = []
        self.process = subprocess.Popen(
            [
                'nvidia-smi',
                '--query-gpu=memory.used,power.draw',
                '--format=csv,noheader,nounits',
                '-lms', str(self.interval_ms),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 2:
                continue
            try:
                self.samples.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.process = None
        self.thread = None

    def peak_mib(self):
        return max((mib for mib, _ in self.samples), default=None)

    def peak_watts(self):
        return max((watts for _, watts in self.samples), default=None)


# Compatibility for benchmark clients that import the old public helper name.
VramSampler = BoardVramSampler


def gpu_now():
    '''One immediate reading, for the idle preflight and per-arm baseline.'''
    completed = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.used,power.draw,clocks.sm,power.limit',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True, text=True, timeout=30,
    )
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise BenchError('nvidia-smi returned no reading: %s' % completed.stderr.strip())
    used, watts, clocks, power_limit = [
        float(part.strip()) for part in lines[0].split(',')
    ]
    return {
        'memory_used_mib': used,
        'power_w': watts,
        'clocks_sm_mhz': clocks,
        'power_limit_w': power_limit,
    }


def process_sampler_for(args):
    '''Dedicated-memory sampler for the ComfyUI server process.'''
    if args.no_process_vram:
        return None
    try:
        from process_vram import ProcessVramSampler, listener_pid
    except ImportError as error:
        raise BenchError(
            'process VRAM sampling is required for README results (%s); '
            'install pywin32/psutil or pass --no-process-vram for a '
            'non-publishable diagnostic run' % error
        ) from error
    pid = args.server_pid
    if pid is None:
        pid = listener_pid(urlparse(args.server).port or 80)
    if pid is None:
        raise BenchError(
            'cannot find the local ComfyUI listener PID; pass --server-pid'
        )
    print(
        'server pid %d: sampling dedicated GPU memory every %d ms'
        % (pid, args.sample_ms),
        flush=True,
    )
    return ProcessVramSampler(pid, interval_ms=args.sample_ms)


# ---------------------------------------------------------------------------
# Prompt execution
# ---------------------------------------------------------------------------

async def run_prompt(session, server, client_id, graph, expected_steps, timeout):
    '''Queue one prompt and time each sampler step from the progress stream.'''
    import aiohttp

    ws_url = server.replace('https://', 'wss://').replace('http://', 'ws://')
    async with session.ws_connect('%s/ws?clientId=%s' % (ws_url, client_id)) as ws:
        async with session.post(
            '%s/prompt' % server,
            json={'prompt': graph, 'client_id': client_id},
        ) as response:
            body = await response.json()
            if response.status != 200:
                raise BenchError(
                    'prompt rejected (%d):\n%s'
                    % (response.status, json.dumps(body, indent=2))
                )
            prompt_id = body['prompt_id']

        boundaries = []
        cached_nodes = []
        last_value = -1
        started = time.perf_counter()
        deadline = started + timeout

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise BenchError(
                    'timed out after %.0f s waiting for prompt %s' % (timeout, prompt_id)
                )
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                raise BenchError(
                    'timed out after %.0f s waiting for prompt %s' % (timeout, prompt_id)
                )
            if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise BenchError('websocket closed before prompt %s finished' % prompt_id)
            if message.type is not aiohttp.WSMsgType.TEXT:
                continue

            payload = json.loads(message.data)
            kind = payload.get('type')
            data = payload.get('data') or {}
            if data.get('prompt_id') not in (None, prompt_id):
                continue

            if kind == 'progress_state':
                for state in (data.get('nodes') or {}).values():
                    if state.get('max') != expected_steps:
                        continue
                    value = state.get('value')
                    if isinstance(value, int) and value > last_value:
                        last_value = value
                        boundaries.append((value, time.perf_counter()))
            elif kind == 'progress':
                if data.get('max') == expected_steps:
                    value = data.get('value')
                    if isinstance(value, int) and value > last_value:
                        last_value = value
                        boundaries.append((value, time.perf_counter()))
            elif kind == 'execution_cached':
                cached_nodes.extend(
                    str(node_id) for node_id in (data.get('nodes') or [])
                    if str(node_id) not in cached_nodes
                )
            elif kind == 'execution_error':
                raise BenchError(
                    'execution failed in node %s (%s): %s'
                    % (data.get('node_id'), data.get('node_type'),
                       data.get('exception_message'))
                )
            elif kind == 'execution_interrupted':
                raise BenchError('execution interrupted')
            elif kind == 'execution_success':
                break
            elif kind == 'executing' and data.get('node') is None:
                break

        return {
            'prompt_id': prompt_id,
            'boundaries': boundaries,
            'cached_nodes': cached_nodes,
            'wall_seconds': time.perf_counter() - started,
        }


def step_durations(boundaries, warmup):
    '''Milliseconds per sampler step, warmup steps dropped.'''
    ordered = [timestamp for _, timestamp in sorted(boundaries)]
    deltas = [
        (ordered[index] - ordered[index - 1]) * 1000.0
        for index in range(1, len(ordered))
    ]
    return deltas[warmup:], deltas


async def require_command_line_sage(session, server, arms):
    '''Refuse to run the Sage arms unless the server actually selected Sage.

    Both Sage rows rely on ComfyUI's --use-sage-attention rather than a node in
    the graph, so on a server without it they would silently measure whatever
    the default attention is and still be labelled SageAttention.
    '''
    if not any(arm in ('sage', 'sage_memory') for arm in arms):
        return
    async with session.get('%s/api/system_stats' % server) as response:
        if response.status != 200:
            raise BenchError(
                'cannot verify --use-sage-attention: /api/system_stats '
                'returned %d' % response.status
            )
        payload = await response.json()
    argv = (payload.get('system') or {}).get('argv') or []
    if '--use-sage-attention' not in argv:
        raise BenchError(
            'the sage and sage_memory arms require a server started with '
            '--use-sage-attention; restart ComfyUI with that flag or drop '
            'those arms'
        )


async def reset_between_arms(session, server, unload_models, *, free_memory=False):
    '''Reset state that would otherwise make a later arm inherit the prior peak.'''
    if not unload_models and not free_memory:
        return
    async with session.post(
        '%s/free' % server,
        json={
            'unload_models': bool(unload_models),
            'free_memory': bool(free_memory),
        },
    ) as response:
        await response.read()


async def prime_server(session, server, client_id, graph, expected_steps, timeout):
    '''Run one discarded prompt so the first measured arm is not the odd one out.'''
    await run_prompt(session, server, client_id, graph, expected_steps, timeout)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_table(records, args, baseline_arm):
    '''The README-format markdown table, ready to paste after a complete run.'''
    del baseline_arm
    arms, workloads = [], []
    for record in records:
        if record['arm'] not in arms:
            arms.append(record['arm'])
        if record['workload'] not in workloads:
            workloads.append(record['workload'])

    grouped = {}
    for record in records:
        grouped.setdefault((record['arm'], record['workload']), []).append(record)

    def cell(arm, workload):
        runs = grouped.get((arm, workload))
        if not runs:
            return None
        speed_runs = [
            run for run in runs
            if run.get('measurement') == 'speed' and not run.get('error')
        ]
        vram_runs = [
            run for run in runs
            if run.get('measurement') == 'vram' and not run.get('error')
        ]
        speed_errors = ' '.join(
            run.get('error', '') for run in runs
            if run.get('measurement') == 'speed'
        ).lower()
        vram_errors = ' '.join(
            run.get('error', '') for run in runs
            if run.get('measurement') == 'vram'
        ).lower()
        vram_key = 'board_peak_mib' if getattr(args, 'no_process_vram', False) else 'process_peak_mib'
        peaks = [
            run[vram_key] for run in vram_runs
            if run.get(vram_key) is not None
        ]
        return {
            'step_ms': (
                statistics.median([run['median_step_ms'] for run in speed_runs])
                if speed_runs else None
            ),
            'peak_mib': statistics.median(peaks) if peaks else None,
            'speed_error': (
                'out of memory' if 'out of memory' in speed_errors else 'failed'
            ) if speed_errors and not speed_runs else None,
            'vram_error': (
                'out of memory' if 'out of memory' in vram_errors else 'failed'
            ) if vram_errors and not vram_runs else None,
        }

    def projected_duration(step_ms):
        seconds = round(step_ms * args.schedule_steps / 1000.0)
        minutes, seconds = divmod(seconds, 60)
        return '~%dm%02ds' % (minutes, seconds) if minutes else '~%ds' % seconds

    header = ['configuration']
    divider = ['---']
    for label in workloads:
        header += ['%s step' % label, '%s %d-step' % (label, args.schedule_steps), '%s VRAM' % label]
        divider += ['---:', '---:', '---:']

    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(divider) + ' |']
    for arm in arms:
        cells = [ARM_LABELS.get(arm, arm)]
        for label in workloads:
            current = cell(arm, label)
            if current is None:
                cells += ['-', '-', '-']
                continue
            if current['step_ms'] is None:
                cells += [current['speed_error'] or '-', '-']
            else:
                cells += [
                    '%.1f s' % (current['step_ms'] / 1000.0),
                    projected_duration(current['step_ms']),
                ]
            cells.append(
                '%.0f MiB' % current['peak_mib']
                if current['peak_mib'] is not None
                else current['vram_error'] or '-'
            )
        lines.append('| ' + ' | '.join(cells) + ' |')

    note = (
        '\nMedian of %d measured steps after %d warmup step(s), on the first %d '
        'steps of a %d-step %s/%s trajectory at %dx%d. Every arm forces '
        'ConvRot QKV CUTLASS config 0. Speed uses stock AIMDO residency; VRAM '
        'uses a separate first-step run with AIMDO residency fixed to 0 blocks. VRAM is '
        '%s.%s'
        % (
            args.measure_steps, args.warmup_steps,
            args.warmup_steps + args.measure_steps + 1, args.schedule_steps,
            args.sampler, args.scheduler, args.width, args.height,
            (
                'whole-card nvidia-smi memory (diagnostic only; not README-compatible)'
                if getattr(args, 'no_process_vram', False)
                else "the ComfyUI process's dedicated GPU memory"
            ),
            (
                ''
                if getattr(args, 'no_process_vram', False)
                else ' Executor, model, and allocator state is reset between arms'
            ),
        )
    )
    return '\n'.join(lines) + '\n' + note


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare H3 attention/memory configurations end to end '
                    'via the ComfyUI prompt API.'
    )
    parser.add_argument('--server', default=DEFAULT_SERVER)
    parser.add_argument(
        '--arms', default=DEFAULT_ARMS,
        help='comma-separated, run in this order; choices: %s' % ', '.join(ARMS),
    )
    parser.add_argument(
        '--workloads', default=','.join(WORKLOADS),
        help='comma-separated durations; choices: %s' % ', '.join(WORKLOADS),
    )
    parser.add_argument(
        '--baseline', default='',
        help='arm used as the speedup denominator (default: the first arm)',
    )
    parser.add_argument('--width', type=int, default=ONE_MP[0])
    parser.add_argument('--height', type=int, default=ONE_MP[1])
    parser.add_argument(
        '--schedule-steps', type=int, default=20,
        help='true schedule length; defines the sigma trajectory',
    )
    parser.add_argument('--measure-steps', type=int, default=3,
                        help='steps included in the per-arm median')
    parser.add_argument('--warmup-steps', type=int, default=1,
                        help='leading steps discarded')
    parser.add_argument('--repeat', type=int, default=1,
                        help='times to run the whole matrix')
    parser.add_argument('--sampler', default='res_multistep')
    parser.add_argument('--scheduler', default='simple')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument(
        '--unique-arm-seeds', action='store_true',
        help='use a distinct noise seed for every measured arm; this forbids '
             'cross-arm sampler cache hits but gives arms different noise',
    )
    parser.add_argument('--unet', default=DEFAULT_UNET)
    parser.add_argument(
        '--conditioning-tokens', type=int,
        default=DEFAULT_CONDITIONING_TOKENS,
        help='token count of the exact cached Qwen conditioning entry',
    )
    parser.add_argument(
        '--conditioning-cache-entry',
        default=DEFAULT_CONDITIONING_CACHE_ENTRY,
        help='exact h3_cond_cache filename; must contain [1, tokens, 5120] FP32',
    )
    parser.add_argument(
        '--allow-zero-conditioning', action='store_true',
        default=DEFAULT_ALLOW_ZERO_CONDITIONING,
        help='allow an empty --conditioning-cache-entry to use zero Qwen states; '
             'not representative enough for README results',
    )
    parser.add_argument('--timeout', type=float, default=1800.0,
                        help='per-prompt seconds')
    parser.add_argument('--settle-seconds', type=float, default=6.0,
                        help='idle wait after /free before each arm')
    parser.add_argument(
        '--no-prime', dest='prime', action='store_false',
        help='skip the discarded priming run; the first arm then absorbs the '
             'checkpoint load',
    )
    parser.add_argument(
        '--unload-between-arms', action='store_true',
        help='unload model weights between arms for a cleaner VRAM floor',
    )
    parser.add_argument('--sample-ms', type=int, default=50,
                        help='dedicated process-memory sampling interval')
    parser.add_argument(
        '--server-pid', type=int, default=None,
        help='ComfyUI process to sample; default is the --server port listener',
    )
    parser.add_argument(
        '--no-process-vram', action='store_true',
        help='use whole-card nvidia-smi memory for diagnostics; results are not '
             'compatible with the README VRAM columns',
    )
    parser.add_argument('--idle-watts', type=float, default=60.0,
                        help='preflight refuses above this')
    parser.add_argument(
        '--expected-power-limit-w', type=float,
        default=DEFAULT_EXPECTED_POWER_LIMIT_W,
        help='required board power limit; use 0 to disable this check',
    )
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='JSON checkpoint path (default: %(default)s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='build and print one prompt without sampling')
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)

    args.arm_list = [name.strip() for name in args.arms.split(',') if name.strip()]
    unknown = [name for name in args.arm_list if name not in ARMS]
    if unknown:
        parser.error('unknown arm(s): %s' % ', '.join(unknown))
    if not args.arm_list:
        parser.error('no arms selected')

    args.workload_list = [
        name.strip() for name in args.workloads.split(',') if name.strip()
    ]
    unknown = [name for name in args.workload_list if name not in WORKLOADS]
    if unknown:
        parser.error('unknown workload(s): %s' % ', '.join(unknown))
    if not args.workload_list:
        parser.error('no workloads selected')

    if args.measure_steps < 1:
        parser.error('need at least one measured step')
    if args.warmup_steps < 0:
        parser.error('--warmup-steps must be non-negative')
    if args.repeat < 1:
        parser.error('--repeat must be at least 1')
    if args.conditioning_tokens < 1:
        parser.error('--conditioning-tokens must be at least 1')
    if not args.conditioning_cache_entry and not args.allow_zero_conditioning:
        parser.error(
            '--conditioning-cache-entry is required for representative results; '
            'pass --allow-zero-conditioning only for diagnostics'
        )
    if args.sample_ms < 20:
        parser.error('--sample-ms must be at least 20')
    if args.expected_power_limit_w < 0:
        parser.error('--expected-power-limit-w must be non-negative')
    if not args.baseline:
        args.baseline = args.arm_list[0]
    if args.baseline not in args.arm_list:
        parser.error('--baseline %r is not among the selected arms' % args.baseline)
    if not args.dry_run and not args.i_understand_this_uses_gpu:
        parser.error(
            'this drives real sampling on the GPU; pass '
            '--i-understand-this-uses-gpu after confirming the card is idle, '
            'or use --dry-run'
        )
    return args


def measured_seed(args, iteration, arm_index):
    if args.unique_arm_seeds:
        return args.seed + iteration * len(args.arm_list) + arm_index
    return args.seed + iteration


def source_provenance():
    '''Identify the Python and packaged native code behind the result file.'''
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        completed = subprocess.run(
            ['git', '-C', str(root), *arguments],
            capture_output=True, text=True, timeout=30,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    native_bin = root / 'native' / 'bin'
    build_id_path = native_bin / 'BUILD_ID'
    git_status = git('status', '--porcelain')
    artifact_hashes = {}
    for name in ('h3_int8_attention_v5.dll', 'libh3_int8_attention.so'):
        path = native_bin / name
        if path.is_file():
            artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'h3_optimizations_head': git('rev-parse', 'HEAD'),
        'h3_optimizations_dirty': bool(git_status),
        'h3_optimizations_dirty_paths': (
            [] if not git_status else git_status.splitlines()
        ),
        'native_build_id': (
            build_id_path.read_text(encoding='utf-8').strip()
            if build_id_path.is_file() else None
        ),
        'native_artifact_sha256': artifact_hashes,
        'benchmark_script': str(Path(__file__).resolve()),
    }


def write_output(path_text, args, records):
    '''Persist after every arm so an OOM does not lose completed results.'''
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 3,
        'memory_metric': (
            'whole-card nvidia-smi memory; diagnostic only'
            if args.no_process_vram
            else 'ComfyUI process dedicated GPU memory'
        ),
        'config': {
            key: value for key, value in vars(args).items()
            if key not in ('i_understand_this_uses_gpu', 'dry_run', 'provenance')
        },
        'provenance': args.provenance,
        'records': records,
        'readme_table': render_table(records, args, args.baseline),
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


async def run_matrix(args):
    import aiohttp

    client_id = str(uuid.uuid4())
    records = []
    args.provenance = source_provenance()

    async with aiohttp.ClientSession() as session:
        await require_command_line_sage(session, args.server, args.arm_list)
        schemas = Schemas(args.server, session)

        if args.dry_run:
            graph, executed = await build_prompt(
                schemas, args.arm_list[0], WORKLOADS[args.workload_list[0]], args,
                measurement='speed',
            )
            print(json.dumps(graph, indent=2))
            print(
                '\n# arm=%s workload=%s frames=%d executed_steps=%d'
                % (args.arm_list[0], args.workload_list[0],
                   WORKLOADS[args.workload_list[0]], executed),
                file=sys.stderr,
            )
            return records

        process_sampler = process_sampler_for(args)

        if args.prime:
            prime_arm, prime_label = args.arm_list[0], args.workload_list[0]
            print(
                '=== priming (%s / %s, discarded) ===' % (prime_arm, prime_label),
                flush=True,
            )
            graph, executed = await build_prompt(
                schemas, prime_arm, WORKLOADS[prime_label], args,
                seed=args.seed - 1,
                measurement='speed',
            )
            await prime_server(
                session, args.server, client_id, graph, executed, args.timeout
            )

        for iteration in range(args.repeat):
            for label in args.workload_list:
                frames = WORKLOADS[label]
                for arm_index, arm in enumerate(args.arm_list):
                    iteration_seed = measured_seed(args, iteration, arm_index)
                    for measurement in ('vram', 'speed'):
                        tag = '%s / %s / %s' % (arm, label, measurement)
                        if args.repeat > 1:
                            tag += ' (run %d)' % (iteration + 1)
                        print('=== %s ===' % tag, flush=True)

                        graph, executed = await build_prompt(
                            schemas, arm, frames, args, seed=iteration_seed,
                            measurement=measurement,
                        )
                        await reset_between_arms(
                            session,
                            args.server,
                            args.unload_between_arms or measurement == 'vram',
                            free_memory=not args.no_process_vram,
                        )
                        if args.unload_between_arms or not args.no_process_vram:
                            await asyncio.sleep(args.settle_seconds)
                        baseline_gpu = gpu_now()
                        sample_process_memory = (
                            measurement == 'vram' and process_sampler is not None
                        )
                        process_baseline_mib = (
                            process_sampler.read_once() / (1024.0 * 1024.0)
                            if sample_process_memory else None
                        )

                        board_sampler = BoardVramSampler(interval_ms=args.sample_ms)
                        board_sampler.start()
                        if sample_process_memory:
                            process_sampler.start()
                        record = {
                            'arm': arm, 'workload': label,
                            'measurement': measurement,
                            'residency': MEASUREMENT_RESIDENCY[measurement],
                            'iteration': iteration + 1,
                            'frames': frames, 'seconds': round(frames / FPS, 2),
                            'width': args.width, 'height': args.height,
                            'executed_steps': executed,
                            'baseline_gpu': baseline_gpu,
                            'process_baseline_mib': process_baseline_mib,
                            'seed': iteration_seed,
                            'process_peak_mib': None,
                            'board_peak_mib': None,
                        }
                        try:
                            result = await run_prompt(
                                session, args.server, client_id, graph, executed,
                                args.timeout,
                            )
                            record['cached_nodes'] = result['cached_nodes']
                            if 'sample' in result['cached_nodes']:
                                raise BenchError(
                                    'sampler node was served from execution cache '
                                    'for prompt %s' % result['prompt_id']
                                )
                            record.update({
                                'wall_seconds': result['wall_seconds'],
                                'prompt_id': result['prompt_id'],
                            })
                            if measurement == 'speed':
                                measured, every = step_durations(
                                    result['boundaries'], args.warmup_steps
                                )
                                if not measured:
                                    raise BenchError(
                                        'no measured steps; the server reported %d step '
                                        'boundaries. A count of zero means the prompt '
                                        'was served from the execution cache rather '
                                        'than run, so this graph had already been '
                                        'executed with seed %d.'
                                        % (len(result['boundaries']), iteration_seed)
                                    )
                                median_ms = statistics.median(measured)
                                record.update({
                                    'median_step_ms': median_ms,
                                    'min_step_ms': min(measured),
                                    'measured_step_ms': measured,
                                    'all_step_ms': every,
                                    'extrapolated_full_run_s':
                                        median_ms * args.schedule_steps / 1000.0,
                                })
                        except BenchError as error:
                            record['error'] = str(error)
                            print('  FAILED: %s' % error, flush=True)
                        finally:
                            if sample_process_memory:
                                process_sampler.stop()
                            board_sampler.stop()

                        if measurement == 'vram':
                            record['process_peak_mib'] = (
                                None if process_sampler is None
                                else process_sampler.peak_mib()
                            )
                        record['board_peak_mib'] = board_sampler.peak_mib()
                        record['peak_watts'] = board_sampler.peak_watts()
                        record['process_peak_over_baseline_mib'] = (
                            None
                            if record['process_peak_mib'] is None
                            else record['process_peak_mib'] - process_baseline_mib
                        )
                        record['board_peak_over_baseline_mib'] = (
                            None if record['board_peak_mib'] is None
                            else record['board_peak_mib'] - baseline_gpu['memory_used_mib']
                        )
                        if (
                            measurement == 'vram'
                            and process_sampler is not None
                            and record['process_peak_mib'] is None
                            and not record.get('error')
                        ):
                            record['error'] = (
                                'process VRAM sampler returned no readings; '
                                'the run is not publishable'
                            )
                            print('  FAILED: %s' % record['error'], flush=True)
                        records.append(record)
                        write_output(args.output, args, records)

                        if not record.get('error'):
                            if measurement == 'speed':
                                print(
                                    '  %.1f s/step (median of %d), board peak %.0f MiB, '
                                    'full %d-step run ~%.0f s'
                                    % (
                                        record['median_step_ms'] / 1000.0,
                                        len(record['measured_step_ms']),
                                        record['board_peak_mib'] or 0.0,
                                        args.schedule_steps,
                                        record['extrapolated_full_run_s'],
                                    ),
                                    flush=True,
                                )
                            else:
                                peak_mib = (
                                    record['board_peak_mib']
                                    if args.no_process_vram
                                    else record['process_peak_mib']
                                )
                                print(
                                    '  process peak %s, board peak %.0f MiB'
                                    % (
                                        'disabled' if args.no_process_vram
                                        else '%.0f MiB' % peak_mib,
                                        record['board_peak_mib'] or 0.0,
                                    ),
                                    flush=True,
                                )

        await reset_between_arms(
            session,
            args.server,
            args.unload_between_arms,
            free_memory=not args.no_process_vram,
        )

    return records


def main(argv=None):
    args = parse_args(argv)

    if not args.dry_run:
        idle = gpu_now()
        print(
            'GPU preflight: %.0f MiB used, %.1f W draw, %.0f MHz, %.0f W limit'
            % (
                idle['memory_used_mib'], idle['power_w'],
                idle['clocks_sm_mhz'], idle['power_limit_w'],
            )
        )
        if idle['power_w'] > args.idle_watts:
            raise SystemExit(
                'GPU is drawing %.1f W, above the --idle-watts limit of %.1f; '
                'something is already running. Stop it, or raise the limit '
                'deliberately.' % (idle['power_w'], args.idle_watts)
            )
        if (
            args.expected_power_limit_w
            and abs(idle['power_limit_w'] - args.expected_power_limit_w) > 1.0
        ):
            raise SystemExit(
                'GPU power limit is %.0f W, expected %.0f W for README '
                'comparability; set the limit first, or override '
                '--expected-power-limit-w deliberately.'
                % (idle['power_limit_w'], args.expected_power_limit_w)
            )

    try:
        records = asyncio.run(run_matrix(args))
    except BenchError as error:
        raise SystemExit(str(error))

    if args.dry_run:
        return 0

    successful = [record for record in records if not record.get('error')]
    if successful:
        print('\n' + render_table(records, args, args.baseline))
    else:
        print('\nno arm completed; nothing to tabulate')

    if args.output:
        write_output(args.output, args, records)
        print('\nwrote %s' % args.output)

    return 0 if successful else 1


if __name__ == '__main__':
    sys.exit(main())
