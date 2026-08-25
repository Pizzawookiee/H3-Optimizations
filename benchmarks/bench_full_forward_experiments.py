'''Run FinalLayer and streamed Kitchen experiments through complete H3 forwards.

Every arm uses H3 Memory Optimization, explicit Kitchen INT8 sparse attention,
and H3 AIMDO Residency Limiter at 0 blocks. The experiment nodes are available
only in the temporary server launched by this script.
'''

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACK_ROOT.parents[1]
for _root in (str(Path(__file__).resolve().parent), str(PACK_ROOT), str(COMFY_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

import bench_attention_arms as benchmark  # noqa: E402


LOG_DIGEST_PREFIX = 'H3_FULL_FORWARD_DIGEST='
PRODUCTION_ALLOC_CONF = (
    'backend:native,garbage_collection_threshold:0.95,expandable_segments:False'
)

BASE_CHAIN = [
    ('H3MemoryOptimization', {}),
    (
        'H3SparseAttentionAdvanced',
        {
            'video_budget': 0.3,
            'early_steps': 0,
            'early_kv': 0.3,
            'late_steps': 0,
            'late_kv': 0.3,
            'backend': 'Kitchen INT8',
        },
    ),
]

benchmark.WORKLOADS = {'5s': 124}
benchmark.DEFAULT_PROMPT = ''
benchmark.ARMS = {
    'baseline': BASE_CHAIN + [
        ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
    ],
    'final_layer_chunking': BASE_CHAIN + [
        (
            'H3FullForwardExperiment',
            {'variant': 'final_layer_chunking', 'chunk_rows': 4096},
        ),
        ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
    ],
    'streamed_kitchen_output': BASE_CHAIN + [
        (
            'H3FullForwardExperiment',
            {'variant': 'streamed_kitchen_output', 'chunk_rows': 4096},
        ),
        ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
    ],
    'combined': BASE_CHAIN + [
        (
            'H3FullForwardExperiment',
            {'variant': 'combined', 'chunk_rows': 4096},
        ),
        ('H3AIMDOResidencyLimiter', {'residency': '0 blocks'}),
    ],
}
benchmark.DEFAULT_ARMS = ','.join(benchmark.ARMS)
benchmark.ARM_LABELS = {
    'baseline': 'Kitchen INT8 + AIMDO 0 (baseline)',
    'final_layer_chunking': 'FinalLayer chunking + AIMDO 0',
    'streamed_kitchen_output': 'Streamed Kitchen output + AIMDO 0',
    'combined': 'FinalLayer chunking + streamed Kitchen output + AIMDO 0',
}

_build_prompt = benchmark.build_prompt


async def build_prompt(schemas, arm, frames, args, seed=None):
    graph, executed = await _build_prompt(
        schemas, arm, frames, args, seed=seed
    )
    effective_seed = args.seed if seed is None else seed
    graph['sink'] = {
        'class_type': 'H3FullForwardDigest',
        'inputs': await schemas.inputs(
            'H3FullForwardDigest',
            {
                'arm': arm,
                'reference_key': '%d:%d' % (frames, effective_seed),
            },
            {'samples': ['sample', 0]},
        ),
        '_meta': {'title': 'full_forward_digest'},
    }
    return graph, executed


benchmark.build_prompt = build_prompt


def _launcher_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--launch-server', action='store_true')
    parser.add_argument('--launch-port', type=int, default=8190)
    parser.add_argument('--server-root', default='')
    parser.add_argument('--server-log', default='')
    parser.add_argument('--extra-model-paths-config', default='')
    parser.add_argument('--max-full-forward-abs', type=float, default=1.0)
    parser.add_argument('--max-full-forward-relative-rmse', type=float, default=1e-3)
    return parser.parse_known_args(argv)


def _request_json(url, payload=None, timeout=10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def _wait_for_server(server, process, timeout=180):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('temporary ComfyUI server exited with %d' % process.returncode)
        try:
            system = _request_json('%s/system_stats' % server)
            schema = _request_json(
                '%s/object_info/H3FullForwardExperiment' % server
            )
            node = schema.get('H3FullForwardExperiment')
            if node is not None:
                return {
                    'system_stats': system,
                    'experiment_node_module': node.get('python_module'),
                    'server': server,
                }
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(0.5)
    raise RuntimeError('temporary ComfyUI server did not become ready: %s' % last_error)


def _option_value(argv, name):
    try:
        index = argv.index(name)
    except ValueError:
        return ''
    return argv[index + 1] if index + 1 < len(argv) else ''


def _attach_server_evidence(
    output_path,
    log_path,
    identity,
    *,
    max_abs,
    max_relative_rmse,
):
    if not output_path:
        return []
    output = Path(output_path)
    if not output.is_file():
        return ['benchmark output was not written']
    log_text = Path(log_path).read_text(encoding='utf-8', errors='replace')
    all_digests = []
    for line in log_text.splitlines():
        marker = line.find(LOG_DIGEST_PREFIX)
        if marker >= 0:
            all_digests.append(json.loads(line[marker + len(LOG_DIGEST_PREFIX):]))
    payload = json.loads(output.read_text(encoding='utf-8'))
    records = [record for record in payload.get('records', []) if not record.get('error')]
    measured_keys = {
        (record['arm'], '%d:%d' % (record['frames'], record['seed']))
        for record in records
    }
    digests = [
        digest for digest in all_digests
        if (digest['arm'], digest['reference_key']) in measured_keys
    ]
    by_arm = {record['arm']: record for record in records}
    digest_by_arm = {record['arm']: record for record in digests}
    failures = []
    if len(digests) != len(records):
        failures.append(
            'captured %d output digests for %d successful arms'
            % (len(digests), len(records))
        )
    aimdo_zero_count = log_text.count(
        'AIMDO residency limited to 0 block-equivalent(s), 0 pages, 0 MiB'
    )
    if aimdo_zero_count < len(records):
        failures.append(
            'AIMDO zero-page limiter applied %d times for %d arms'
            % (aimdo_zero_count, len(records))
        )
    final_layer_install_count = log_text.count(
        '[H3 Optimizations] patched FinalLayer: chunk_rows=4096'
    )
    streamed_default_count = sum(
        'H3 Optimizations] armed:' in line and 'streamed_out' in line
        for line in log_text.splitlines()
    )
    if 'baseline' in by_arm and final_layer_install_count < 1:
        failures.append('production baseline did not install chunked FinalLayer')
    if 'baseline' in by_arm and streamed_default_count < 1:
        failures.append('production baseline did not select streamed Kitchen output')

    for arm, record in by_arm.items():
        digest = digest_by_arm.get(arm)
        if digest is None:
            failures.append('%s has no output digest' % arm)
            continue
        executed = int(record['executed_steps'])
        execution = digest.get('execution', {})
        if arm in ('final_layer_chunking', 'combined'):
            if int(execution.get('final_layer_calls', -1)) != executed:
                failures.append('%s did not execute chunked FinalLayer %d times' % (arm, executed))
            comparison = digest['comparison']
            if comparison['max_abs'] > max_abs:
                failures.append('%s max abs %.6g exceeds %.6g' % (arm, comparison['max_abs'], max_abs))
            if comparison['relative_rmse'] > max_relative_rmse:
                failures.append(
                    '%s relative RMSE %.6g exceeds %.6g'
                    % (arm, comparison['relative_rmse'], max_relative_rmse)
                )
        if arm in ('streamed_kitchen_output', 'combined'):
            layers = int(execution.get('attention_layers', -1))
            if int(execution.get('streamed_blocks', -1)) != layers * executed:
                failures.append('%s did not stream every attention block' % arm)
            if int(execution.get('streamed_complete_forwards', -1)) != executed:
                failures.append('%s did not complete %d streamed forwards' % (arm, executed))
            if execution.get('route_q_tile') != 64 or execution.get('route_kv_tile') != 64:
                failures.append('%s did not use the explicit 64Q x 64KV Kitchen route' % arm)

    streamed = digest_by_arm.get('streamed_kitchen_output')
    if streamed is not None and not streamed['comparison']['exact']:
        failures.append('streamed Kitchen output changed the full sampler latent')
    final = digest_by_arm.get('final_layer_chunking')
    combined = digest_by_arm.get('combined')
    if final is not None and combined is not None:
        if final['streams'] != combined['streams']:
            failures.append('combined output differs from FinalLayer-only output')

    payload['full_forward_validation'] = {
        'server_identity': identity,
        'digests': digests,
        'aimdo_zero_apply_count': aimdo_zero_count,
        'final_layer_install_count': final_layer_install_count,
        'streamed_default_count': streamed_default_count,
        'max_abs_limit': float(max_abs),
        'max_relative_rmse_limit': float(max_relative_rmse),
        'passed': not failures,
        'failures': failures,
        'streamed_complete_forward_logs': log_text.count(
            'streamed Kitchen output completed forward'
        ),
        'chunked_final_layer_forward_logs': log_text.count(
            'chunked FinalLayer completed forward'
        ),
        'server_log': str(Path(log_path).resolve()),
    }
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return failures


def _launch_and_run(launcher, benchmark_argv):
    root = Path(launcher.server_root).resolve()
    log_path = Path(launcher.server_log).resolve()
    required = [root, root / 'user', root / 'input', root / 'output', root / 'temp']
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SystemExit(
            'create the benchmark server directories with PowerShell first: %s'
            % ', '.join(missing)
        )
    if not log_path.parent.is_dir():
        raise SystemExit('server log directory does not exist: %s' % log_path.parent)
    model_paths = Path(launcher.extra_model_paths_config).resolve()
    if not model_paths.is_file():
        raise SystemExit('extra model paths file does not exist: %s' % model_paths)

    server = 'http://127.0.0.1:%d' % launcher.launch_port
    command = [
        sys.executable,
        '-s',
        str(COMFY_ROOT / 'main.py'),
        '--listen', '127.0.0.1',
        '--port', str(launcher.launch_port),
        '--user-directory', str(root / 'user'),
        '--input-directory', str(root / 'input'),
        '--output-directory', str(root / 'output'),
        '--temp-directory', str(root / 'temp'),
        '--extra-model-paths-config', str(model_paths),
        '--disable-all-custom-nodes',
        '--whitelist-custom-nodes', 'H3-Optimizations', 'comfyui-kjnodes',
        '--disable-cuda-malloc',
        '--disable-auto-launch',
        '--disable-metadata',
    ]
    environment = os.environ.copy()
    environment['H3_OPTIMIZATIONS_BENCHMARK_NODES'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'
    environment.setdefault('PYTORCH_CUDA_ALLOC_CONF', PRODUCTION_ALLOC_CONF)

    identity = None
    with log_path.open('w', encoding='utf-8') as server_log:
        process = subprocess.Popen(
            command,
            cwd=COMFY_ROOT,
            env=environment,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            identity = _wait_for_server(server, process)
            result = benchmark.main([*benchmark_argv, '--server', server])
        finally:
            try:
                _request_json(
                    '%s/free' % server,
                    {'unload_models': True, 'free_memory': True},
                    timeout=30,
                )
            except Exception:
                pass
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)

    failures = _attach_server_evidence(
        _option_value(benchmark_argv, '--output'),
        log_path,
        identity,
        max_abs=launcher.max_full_forward_abs,
        max_relative_rmse=launcher.max_full_forward_relative_rmse,
    )
    if failures:
        print(
            'full-forward validation failed: %s' % '; '.join(failures),
            file=sys.stderr,
        )
        return 1
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    launcher, remaining = _launcher_args(argv)
    if not launcher.launch_server:
        return benchmark.main(remaining)
    if not launcher.server_root or not launcher.server_log:
        raise SystemExit('--launch-server requires --server-root and --server-log')
    if not launcher.extra_model_paths_config:
        raise SystemExit('--launch-server requires --extra-model-paths-config')
    return _launch_and_run(launcher, remaining)


if __name__ == '__main__':
    raise SystemExit(main())
