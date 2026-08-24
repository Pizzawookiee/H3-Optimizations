'''End-to-end comparison of H3 attention configurations through the ComfyUI prompt API.

Unlike the kernel benchmarks in this directory, which time one attention call on
synthetic already-projected Q/K/V, this drives the *running* ComfyUI server with
real prompts: real checkpoint, real conditioning, real sampler. What it reports
is therefore what a user actually experiences, including QKV projection, MLP,
and every allocator effect the kernel benchmarks deliberately exclude.

Only a few steps are executed per arm. A full 20-step H3 run at 1 MP costs
minutes per arm, and the sampler's per-step cost is essentially flat across the
trajectory, so a warmup step plus a small measured series buys the same per-step
number for a fraction of the GPU time. The schedule is still built at its true
length and then truncated with SplitSigmas, so the executed steps sit on the
real sigma trajectory rather than on a compressed one.

Two numbers per cell:

  step ms   median wall time of one sampler step, warmup steps discarded
  peak MiB  highest driver-level VRAM seen during the run, from nvidia-smi

The VRAM figure is driver-level and whole-GPU, not this process's torch
allocator. It is the number that decides whether a run fits on the card, which
is what the comparison is for, but it does include anything else resident on
the GPU. The idle baseline captured before each arm is reported next to it.

Two things keep that figure meaningful. A discarded priming run loads the
checkpoint and populates the conditioning cache before the first measured arm,
and nothing between arms is allowed to reset the server's executor. Resetting it
evicts the cached conditioning, and the next arm then pulls the 32B text encoder
back onto the card alongside the resident diffusion model -- a large constant
added to every arm alike, which drowns out the difference being measured. If
every arm reports nearly the same peak, suspect that before believing the
backends are equivalent.

Run from the ComfyUI root against an already-running server, for example:

    .\\.venv\\Scripts\\python.exe custom_nodes\\H3-Optimizations\\benchmarks\\bench_attention_arms.py --i-understand-this-uses-gpu

This script never imports torch and never touches CUDA in-process. All GPU work
happens in the ComfyUI server it talks to.
'''

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path


DEFAULT_SERVER = 'http://127.0.0.1:8188'

DEFAULT_UNET = r'hf_minimax_h3\minimax_h3_fl2va_pruned_int8_convrot.safetensors'
DEFAULT_CLIP = r'hf_minimax_h3\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
DEFAULT_VAE = r'hf_minimax_h3\minimax_h3_video_vae_fp16.safetensors'

# 16:9 at 1.0 megapixels, taken from the resolution table shipped in ComfyUI's
# MiniMax H3 template workflow. Both dimensions are the required multiple of 32.
ONE_MP = (1376, 768)

# H3 runs at 24 fps and snaps frame counts up to a 17k+5 grid, so neither
# duration lands exactly on its nominal seconds. The grid value is what the
# model actually samples, so that is what gets reported.
FPS = 24
WORKLOADS = {
    '5s': 124,   # 5.17 s
    '10s': 243,  # 10.13 s
}

DEFAULT_PROMPT = (
    'A slow cinematic dolly shot through a rain-slicked neon alley at night, '
    'steam rising from vents, reflections shimmering on wet asphalt.'
)

# Each arm is the chain of model-patch nodes inserted between the loader and the
# guider. An empty chain is the unpatched model exactly as the checkpoint loads.
#
# The last three form a ladder in which each rung changes exactly one thing, so
# a speedup can be attributed instead of merely observed:
#
#   kitchen      -> h3opt_dense   adds chunked QKV and bounded MLP, attention held
#                                 at ComfyUI's public dense comfy_kitchen_int8
#   h3opt_dense  -> h3opt_kv100   swaps that dense backend for the pack's own
#                                 native Kitchen INT8 kernel, still at full density
#   h3opt_kv100  -> h3opt         drops video KV density to 0.3, same kernel
#
# The 1.0 rung matters because H3SparseAttention at full budget still executes
# through the sparse backend. Comparing 0.3 against the resolved dense path
# instead would change the kernel and the density together and attribute both to
# sparsity.
ARMS = {
    'stock': [],
    'pytorch': [
        ('ModelAttentionBackend', {'attention': 'pytorch attention'}),
    ],
    'kitchen': [
        ('ModelAttentionBackend', {'attention': 'comfy kitchen attention'}),
    ],
    'sage': [
        ('PathchSageAttentionKJ', {'sage_attention': 'auto', 'allow_compile': False}),
    ],
    'h3opt_dense': [
        ('H3MemoryOptimization', {}),
    ],
    'h3opt_kv100': [
        ('H3MemoryOptimization', {}),
        ('H3SparseAttention', {'video_budget': 1.0, 'denser_early_late_steps': False}),
    ],
    'h3opt': [
        ('H3MemoryOptimization', {}),
        ('H3SparseAttention', {'video_budget': 0.3, 'denser_early_late_steps': False}),
    ],
}

DEFAULT_ARMS = 'kitchen,sage,h3opt_dense,h3opt_kv100,h3opt'

# What each arm is called in a published table. The internal keys are short
# enough to type on a command line but say nothing about what was configured;
# a reader of the results needs the node names they would actually wire up.
ARM_LABELS = {
    'stock': 'ComfyUI default attention (no pack)',
    'pytorch': 'PyTorch attention (no pack)',
    'kitchen': 'Comfy Kitchen INT8 dense (no pack)',
    'sage': 'SageAttention dense (no pack)',
    'h3opt_dense': 'H3 Memory Optimization',
    'h3opt_kv100': 'H3 Memory Optimization + H3 Sparse Attention (KV 100%)',
    'h3opt': 'H3 Memory Optimization + H3 Sparse Attention (KV 30%, default)',
}


# ---------------------------------------------------------------------------
# Server schema
# ---------------------------------------------------------------------------

class Schemas:
    '''Lazily fetched /object_info, used to fill required inputs with defaults.

    Building prompts against the live schema rather than a hardcoded input list
    means an arm keeps working when a node gains an input. A node that gains a
    *required* input with no default is the one case that still needs a code
    change here, and it raises rather than sending a malformed prompt.
    '''

    def __init__(self, server, session):
        self.server = server
        self.session = session
        self.cache = {}

    async def get(self, node_type):
        if node_type not in self.cache:
            url = '%s/object_info/%s' % (self.server, node_type)
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise BenchError(
                        'server has no node %r (status %d); is the pack installed '
                        'and the server restarted?' % (node_type, response.status)
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
    if isinstance(kind, list):          # inline combo
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

async def build_prompt(schemas, arm, frames, args, seed=None):
    '''The API-format graph for one arm at one duration.

    The sigma schedule is built from the *unpatched* model and truncated with
    SplitSigmas, so every arm samples an identical prefix of an identical
    trajectory and the only difference between arms is how attention executes.

    ``seed`` must differ between the priming run and the measured runs, and
    between repeats. ComfyUI caches node outputs by input, so a graph it has
    already executed is served from cache and never touches the GPU: the arm
    then reports zero step boundaries and no VRAM at all. Arms within one
    iteration deliberately share a seed, since identical noise is what makes
    their timings comparable.
    '''
    # The server reports a progress boundary at values 1..N, never 0, so N
    # executed steps yield only N-1 timed intervals. Ask for one extra step so
    # that --measure-steps survives dropping the warmup.
    executed = args.warmup_steps + args.measure_steps + 1
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
    await node('clip', 'CLIPLoader', {'clip_name': args.clip, 'type': 'minimax'})
    await node('vae', 'VAELoader', {'vae_name': args.vae})

    # No first_frame or last_frame: pure text to video, so the packed layout
    # carries text, audio and video and no reference items.
    await node(
        'cond', 'MiniMaxH3ImageToVideo',
        {'prompt': args.prompt, 'width': args.width, 'height': args.height,
         'length': frames},
        {'clip': ['clip', 0], 'vae': ['vae', 0]},
    )

    model_ref = ['loader', 0]
    for index, (node_type, overrides) in enumerate(ARMS[arm]):
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
            'latent_image': ['cond', 1],
        },
    )
    # PreviewAny is an output node that writes nothing, so the graph terminates
    # without a VAE decode competing for the VRAM being measured.
    await node('sink', 'PreviewAny', {}, {'source': ['sample', 0]})
    return graph, executed


# ---------------------------------------------------------------------------
# VRAM sampling
# ---------------------------------------------------------------------------

class VramSampler:
    '''Driver-level VRAM, streamed from one long-lived nvidia-smi process.'''

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


def gpu_now():
    '''One immediate reading, for the idle preflight and per-arm baseline.'''
    completed = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.used,power.draw,clocks.sm',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True, text=True, timeout=30,
    )
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise BenchError('nvidia-smi returned no reading: %s' % completed.stderr.strip())
    used, watts, clocks = [float(part.strip()) for part in lines[0].split(',')]
    return {'memory_used_mib': used, 'power_w': watts, 'clocks_sm_mhz': clocks}


# ---------------------------------------------------------------------------
# Prompt execution
# ---------------------------------------------------------------------------

async def run_prompt(session, server, client_id, graph, expected_steps, timeout):
    '''Queue one prompt and time each sampler step from the progress stream.

    Returns the timestamp of every step boundary. The interval before the first
    boundary covers model load, text encode and first-step compilation, which is
    exactly why the caller discards warmup steps rather than averaging them in.
    '''
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


async def reset_between_arms(session, server, unload_models):
    '''Release model weights between arms without forcing a text re-encode.

    Never send free_memory here. The server treats that flag as "reset the
    executor", which wipes the cached conditioning and makes the next arm reload
    the 32B text encoder *on top of* the still-resident diffusion model. That
    inflates every arm's VRAM peak by the same large constant and buries the
    difference the benchmark exists to measure.

    unload_models alone unloads the weights but leaves the executor cache
    intact, so the conditioning survives and only the diffusion model reloads.
    The default is to do nothing at all, which is both the fastest option and
    the one that matches how the model sits in a real session.
    '''
    if not unload_models:
        return
    async with session.post(
        '%s/free' % server, json={'unload_models': True, 'free_memory': False},
    ) as response:
        await response.read()


async def prime_server(session, server, client_id, graph, expected_steps, timeout):
    '''Run one discarded prompt so the first measured arm is not the odd one out.

    Without this the first arm pays the checkpoint load and the text encode
    inside its own measurement window, and its VRAM peak carries load transients
    no later arm sees.
    '''
    await run_prompt(session, server, client_id, graph, expected_steps, timeout)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_table(records, args, baseline_arm):
    '''A markdown table, one row per arm, one column group per duration.'''
    arms, workloads = [], []
    for record in records:
        if record['arm'] not in arms:
            arms.append(record['arm'])
        if record['workload'] not in workloads:
            workloads.append(record['workload'])

    # With --repeat, collapse runs of the same cell to their median.
    grouped = {}
    for record in records:
        if record.get('error'):
            continue
        grouped.setdefault((record['arm'], record['workload']), []).append(record)

    def cell(arm, workload):
        runs = grouped.get((arm, workload))
        if not runs:
            return None
        return {
            'step_ms': statistics.median([run['median_step_ms'] for run in runs]),
            'peak_mib': statistics.median(
                [run['peak_mib'] for run in runs if run['peak_mib'] is not None]
            ) if any(run['peak_mib'] is not None for run in runs) else None,
        }

    header = ['arm']
    divider = ['---']
    for label in workloads:
        header += ['%s step ms' % label, '%s peak MiB' % label, '%s vs base' % label]
        divider += ['---:', '---:', '---:']

    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(divider) + ' |']
    for arm in arms:
        cells = [ARM_LABELS.get(arm, arm)]
        for label in workloads:
            current = cell(arm, label)
            base = cell(baseline_arm, label)
            if current is None:
                cells += ['-', '-', '-']
                continue
            ratio = (
                '%.2fx' % (base['step_ms'] / current['step_ms'])
                if base and current['step_ms'] else '-'
            )
            cells += [
                '%.0f' % current['step_ms'],
                '%.0f' % current['peak_mib'] if current['peak_mib'] else '-',
                ratio,
            ]
        lines.append('| ' + ' | '.join(cells) + ' |')

    note = (
        '\nMedian of %d measured steps after %d warmup step(s), on the first %d '
        'steps of a %d-step %s/%s trajectory at %dx%d. Peak MiB is whole-GPU '
        'driver-level VRAM from nvidia-smi, not this process alone. Speedup is '
        'relative to %s.'
        % (
            args.measure_steps, args.warmup_steps,
            args.warmup_steps + args.measure_steps + 1, args.schedule_steps,
            args.sampler, args.scheduler, args.width, args.height,
            ARM_LABELS.get(baseline_arm, baseline_arm),
        )
    )
    return '\n'.join(lines) + '\n' + note


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Compare H3 attention configurations end to end '
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
                        help='steps averaged per arm')
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
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--unet', default=DEFAULT_UNET)
    parser.add_argument('--clip', default=DEFAULT_CLIP)
    parser.add_argument('--vae', default=DEFAULT_VAE)
    parser.add_argument('--timeout', type=float, default=1800.0,
                        help='per-prompt seconds')
    parser.add_argument('--settle-seconds', type=float, default=6.0,
                        help='idle wait after /free before each arm')
    parser.add_argument(
        '--no-prime', dest='prime', action='store_false',
        help='skip the discarded priming run; the first arm then absorbs the '
             'checkpoint load and the text encode',
    )
    parser.add_argument(
        '--unload-between-arms', action='store_true',
        help='unload model weights between arms for a cleaner VRAM floor; the '
             'cached conditioning survives, so no text re-encode',
    )
    parser.add_argument('--idle-watts', type=float, default=60.0,
                        help='preflight refuses above this')
    parser.add_argument('--output', default='',
                        help='write the full JSON record here')
    parser.add_argument('--dry-run', action='store_true',
                        help='build and print one prompt, contact no GPU')
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


def write_output(path_text, args, records):
    '''Persist after every arm.

    A server-side OOM takes the whole process down with it, and an end-of-run
    write loses every completed arm when that happens. It already did once.
    '''
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'config': {
            key: value for key, value in vars(args).items()
            if key not in ('i_understand_this_uses_gpu', 'dry_run')
        },
        'records': records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


async def run_matrix(args):
    import aiohttp

    client_id = str(uuid.uuid4())
    records = []

    async with aiohttp.ClientSession() as session:
        schemas = Schemas(args.server, session)

        if args.dry_run:
            graph, executed = await build_prompt(
                schemas, args.arm_list[0], WORKLOADS[args.workload_list[0]], args
            )
            print(json.dumps(graph, indent=2))
            print(
                '\n# arm=%s workload=%s frames=%d executed_steps=%d'
                % (args.arm_list[0], args.workload_list[0],
                   WORKLOADS[args.workload_list[0]], executed),
                file=sys.stderr,
            )
            return records

        if args.prime:
            prime_arm, prime_label = args.arm_list[0], args.workload_list[0]
            print(
                '=== priming (%s / %s, discarded) ===' % (prime_arm, prime_label),
                flush=True,
            )
            graph, executed = await build_prompt(
                schemas, prime_arm, WORKLOADS[prime_label], args,
                seed=args.seed - 1,
            )
            await prime_server(
                session, args.server, client_id, graph, executed, args.timeout
            )

        for iteration in range(args.repeat):
            for label in args.workload_list:
                frames = WORKLOADS[label]
                for arm_index, arm in enumerate(args.arm_list):
                    tag = '%s / %s' % (arm, label)
                    if args.repeat > 1:
                        tag += ' (run %d)' % (iteration + 1)
                    print('=== %s ===' % tag, flush=True)

                    iteration_seed = measured_seed(args, iteration, arm_index)
                    graph, executed = await build_prompt(
                        schemas, arm, frames, args, seed=iteration_seed
                    )
                    await reset_between_arms(
                        session, args.server, args.unload_between_arms
                    )
                    if args.unload_between_arms:
                        await asyncio.sleep(args.settle_seconds)
                    baseline_gpu = gpu_now()

                    sampler = VramSampler()
                    sampler.start()
                    record = {
                        'arm': arm, 'workload': label, 'iteration': iteration + 1,
                        'frames': frames, 'seconds': round(frames / FPS, 2),
                        'width': args.width, 'height': args.height,
                        'executed_steps': executed,
                        'baseline_gpu': baseline_gpu,
                        'seed': iteration_seed,
                        'peak_mib': None,
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
                            'wall_seconds': result['wall_seconds'],
                            'extrapolated_full_run_s':
                                median_ms * args.schedule_steps / 1000.0,
                            'prompt_id': result['prompt_id'],
                        })
                    except BenchError as error:
                        record['error'] = str(error)
                        print('  FAILED: %s' % error, flush=True)
                    finally:
                        sampler.stop()

                    record['peak_mib'] = sampler.peak_mib()
                    record['peak_watts'] = sampler.peak_watts()
                    record['peak_over_baseline_mib'] = (
                        None if record['peak_mib'] is None
                        else record['peak_mib'] - baseline_gpu['memory_used_mib']
                    )
                    records.append(record)
                    write_output(args.output, args, records)

                    if not record.get('error'):
                        print(
                            '  %.0f ms/step (median of %d), peak %.0f MiB '
                            '(+%.0f over idle), full %d-step run ~%.0f s'
                            % (
                                record['median_step_ms'],
                                len(record['measured_step_ms']),
                                record['peak_mib'] or 0.0,
                                record['peak_over_baseline_mib'] or 0.0,
                                args.schedule_steps,
                                record['extrapolated_full_run_s'],
                            ),
                            flush=True,
                        )

        await reset_between_arms(session, args.server, args.unload_between_arms)

    return records


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
                'GPU is drawing %.1f W, above the --idle-watts limit of %.1f; '
                'something is already running. Stop it, or raise the limit '
                'deliberately.' % (idle['power_w'], args.idle_watts)
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
