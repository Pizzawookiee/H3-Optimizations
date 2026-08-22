'''Run the Stage 1 H3 MLP sharing probe from a local Ref2VA workflow.'''

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid

import aiohttp


REQUIRED_CLASSES = (
    'UNETLoader',
    'VAELoader',
    'CLIPLoader',
    'LoadImage',
    'ZiroScaleImageToNativeCanvas',
    'ResolutionSelector',
    'MiniMaxH3ReferenceToVideoZi',
    'H3MemoryOptimization',
    'H3MLPSharingProbe',
    'H3MLPSharingProbeOutput',
    'H3SparseAttentionAdvanced',
    'RandomNoise',
    'KSamplerSelect',
    'BasicScheduler',
    'BasicGuider',
    'SamplerCustomAdvanced',
)


def _json_request(url, method='GET', payload=None):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    request = Request(
        url,
        data=data,
        method=method,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError('%s %s failed: %s' % (method, url, detail)) from exc


def _node_by_id(workflow, node_id):
    for node in workflow['nodes']:
        if int(node['id']) == int(node_id):
            return node
    raise ValueError('workflow node %s was not found' % node_id)


def _linked_source(workflow, node, input_name):
    matching = [value for value in node.get('inputs', ()) if value['name'] == input_name]
    if len(matching) != 1 or matching[0].get('link') is None:
        raise ValueError('%s.%s must have exactly one link' % (node['type'], input_name))
    link_id = int(matching[0]['link'])
    matching_links = [value for value in workflow['links'] if int(value[0]) == link_id]
    if len(matching_links) != 1:
        raise ValueError('workflow link %d was not found exactly once' % link_id)
    return _node_by_id(workflow, matching_links[0][1])


def _widgets(node):
    values = node.get('widgets_values_named')
    if not isinstance(values, dict):
        raise ValueError('%s requires named widget values' % node['type'])
    return values


def _literal_widgets(node, names):
    values = _widgets(node)
    missing = [name for name in names if name not in values]
    if missing:
        raise ValueError('%s is missing widget %s' % (node['type'], missing[0]))
    return {name: values[name] for name in names}


def _build_prompt(workflow, args):
    ref = next(
        node for node in workflow['nodes']
        if node['type'] == 'MiniMaxH3ReferenceToVideoZi' and int(node.get('mode', 0)) == 0
    )
    clip = _linked_source(workflow, ref, 'clip')
    video_vae = _linked_source(workflow, ref, 'vae')
    audio_vae = _linked_source(workflow, ref, 'audio_vae')
    prompt_node = _linked_source(workflow, ref, 'prompt')
    resolution = _linked_source(workflow, ref, 'width')
    ref_scale = _linked_source(workflow, ref, 'ref_images.ref_image_0')
    ref_image = _linked_source(workflow, ref_scale, 'image')

    unet = next(
        node for node in workflow['nodes']
        if node['type'] == 'UNETLoader' and int(node.get('mode', 0)) == 0
    )
    memory = next(
        node for node in workflow['nodes']
        if node['type'] == 'H3MemoryOptimization' and int(node.get('mode', 0)) == 0
    )
    noise = next(
        node for node in workflow['nodes']
        if node['type'] == 'RandomNoise' and int(node.get('mode', 0)) == 0
    )
    sampler = next(
        node for node in workflow['nodes']
        if node['type'] == 'KSamplerSelect' and int(node.get('mode', 0)) == 0
    )
    scheduler = next(
        node for node in workflow['nodes']
        if node['type'] == 'BasicScheduler' and int(node.get('mode', 0)) == 0
    )

    ref_values = _widgets(ref)
    if 'value' not in _widgets(prompt_node):
        raise ValueError('the linked prompt node has no named value')
    prompt = {
        '1': {
            'class_type': 'UNETLoader',
            'inputs': _literal_widgets(unet, ('unet_name', 'weight_dtype')),
        },
        '2': {
            'class_type': 'H3MemoryOptimization',
            'inputs': {
                'model': ['1', 0],
                **_literal_widgets(memory, ('fused_qkv', 'mlp_memory', 'chunk_rows')),
            },
        },
        '3': {
            'class_type': 'H3MLPSharingProbe',
            'inputs': {
                'model': ['2', 0],
                'enabled': True,
                'layers': args.layers,
                'include_mean_input': True,
                'mean_batch_rows': args.mean_batch_rows,
                'run_tag': args.run_tag,
            },
        },
        '4': {
            'class_type': 'H3SparseAttentionAdvanced',
            'inputs': {
                'model': ['3', 0],
                'video_budget': args.video_budget,
                'early_steps': 0,
                'early_kv': 0.5,
                'late_steps': 0,
                'late_kv': 0.5,
            },
        },
        '5': {
            'class_type': 'CLIPLoader',
            'inputs': _literal_widgets(clip, ('clip_name', 'type', 'device')),
        },
        '6': {
            'class_type': 'VAELoader',
            'inputs': _literal_widgets(video_vae, ('vae_name',)),
        },
        '7': {
            'class_type': 'VAELoader',
            'inputs': _literal_widgets(audio_vae, ('vae_name',)),
        },
        '8': {
            'class_type': 'LoadImage',
            'inputs': _literal_widgets(ref_image, ('image',)),
        },
        '9': {
            'class_type': 'ZiroScaleImageToNativeCanvas',
            'inputs': {
                'image': ['8', 0],
                **_literal_widgets(
                    ref_scale,
                    (
                        'upscale_method',
                        'short_edge',
                        'max_long_edge',
                        'multiple_of',
                        'downscale_only',
                    ),
                ),
            },
        },
        '10': {
            'class_type': 'ResolutionSelector',
            'inputs': _literal_widgets(
                resolution,
                ('aspect_ratio', 'megapixels', 'multiple'),
            ),
        },
        '11': {
            'class_type': 'MiniMaxH3ReferenceToVideoZi',
            'inputs': {
                'clip': ['5', 0],
                'vae': ['6', 0],
                'audio_vae': ['7', 0],
                'prompt': _widgets(prompt_node)['value'],
                'width': ['10', 0],
                'height': ['10', 1],
                'length': int(ref_values['length']),
                'ref_image_size': ref_values['ref_image_size'],
                'cond_cache': ref_values['cond_cache'],
                'ref_images.ref_image_0': ['9', 0],
            },
        },
        '12': {
            'class_type': 'RandomNoise',
            'inputs': _literal_widgets(noise, ('noise_seed',)),
        },
        '13': {
            'class_type': 'KSamplerSelect',
            'inputs': _literal_widgets(sampler, ('sampler_name',)),
        },
        '14': {
            'class_type': 'BasicScheduler',
            'inputs': {
                'model': ['4', 0],
                **_literal_widgets(scheduler, ('scheduler', 'steps', 'denoise')),
            },
        },
        '15': {
            'class_type': 'BasicGuider',
            'inputs': {
                'model': ['4', 0],
                'conditioning': ['11', 0],
            },
        },
        '16': {
            'class_type': 'SamplerCustomAdvanced',
            'inputs': {
                'noise': ['12', 0],
                'guider': ['15', 0],
                'sampler': ['13', 0],
                'sigmas': ['14', 0],
                'latent_image': ['11', 1],
            },
        },
        '17': {
            'class_type': 'H3MLPSharingProbeOutput',
            'inputs': {
                'samples': ['16', 1],
            },
        },
    }
    return prompt


def _gpu_state():
    result = subprocess.run(
        (
            'nvidia-smi',
            '--query-gpu=power.draw,power.limit,clocks.sm,memory.used',
            '--format=csv,noheader,nounits',
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value.strip()) for value in result.stdout.strip().split(',')]
    return {
        'power_watts': values[0],
        'power_limit_watts': values[1],
        'sm_clock_mhz': values[2],
        'memory_used_mib': values[3],
    }


def _check_idle(server, max_power_watts, max_sm_clock_mhz):
    queue = _json_request(server + '/api/queue')
    running = len(queue.get('queue_running', ()))
    pending = len(queue.get('queue_pending', ()))
    if running or pending:
        raise RuntimeError('Comfy queue is not idle: running=%d pending=%d' % (running, pending))
    gpu = _gpu_state()
    if gpu['power_watts'] > max_power_watts or gpu['sm_clock_mhz'] > max_sm_clock_mhz:
        raise RuntimeError('GPU is not idle: %s' % json.dumps(gpu, sort_keys=True))
    return gpu


def _verify_server(server):
    stats = _json_request(server + '/api/system_stats')
    argv = ' '.join(stats['system'].get('argv', ()))
    if 'ComfyUI\\main.py' not in argv and 'ComfyUI/main.py' not in argv:
        raise RuntimeError('the loopback server does not identify this Comfy checkout')
    for class_name in REQUIRED_CLASSES:
        schema = _json_request(server + '/api/object_info/' + class_name)
        if class_name not in schema:
            raise RuntimeError('server is missing required node %s' % class_name)
    return stats['system'].get('comfyui_version')


async def _wait_for_job(server, socket, prompt_id):
    last_progress = None
    while True:
        try:
            message = await socket.receive(timeout=30)
        except asyncio.TimeoutError:
            job = _json_request(server + '/api/jobs/' + prompt_id)
            if job['status'] in ('completed', 'failed', 'cancelled'):
                return job
            continue
        if message.type != aiohttp.WSMsgType.TEXT:
            continue
        event = json.loads(message.data)
        data = event.get('data', {})
        if data.get('prompt_id') != prompt_id:
            continue
        if event.get('type') in ('progress', 'progress_state'):
            progress = (data.get('value'), data.get('max'), data.get('node'))
            if progress != last_progress:
                print(json.dumps({'event': event['type'], 'progress': progress}))
                last_progress = progress
        if event.get('type') in (
            'execution_success',
            'execution_error',
            'execution_interrupted',
        ):
            return _json_request(server + '/api/jobs/' + prompt_id)


async def _submit_and_wait(server, prompt, client_id, args):
    socket_url = server.replace('http://', 'ws://').replace('https://', 'wss://')
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
            socket_url + '/ws?clientId=' + client_id,
            receive_timeout=30,
        ) as socket:
            gpu = _check_idle(
                server,
                args.max_idle_power_watts,
                args.max_idle_sm_clock_mhz,
            )
            accepted = _json_request(
                server + '/api/prompt',
                method='POST',
                payload={'prompt': prompt, 'client_id': client_id},
            )
            prompt_id = accepted['prompt_id']
            print(json.dumps({
                'accepted': prompt_id,
                'gpu_preflight': gpu,
            }, sort_keys=True))
            job = await _wait_for_job(server, socket, prompt_id)
    return prompt_id, job, gpu


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('workflow', type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--run-tag', default='stage1-ref2va-20pct-sparse')
    parser.add_argument('--layers', default='0,1,2,5,10,20,30,40,47,49')
    parser.add_argument('--mean-batch-rows', type=int, default=1024)
    parser.add_argument('--video-budget', type=float, default=0.2)
    parser.add_argument('--max-idle-power-watts', type=float, default=60.0)
    parser.add_argument('--max-idle-sm-clock-mhz', type=float, default=500.0)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu after explicit authorization')

    server = args.server.rstrip('/')
    version = _verify_server(server)
    with args.workflow.open('r', encoding='utf-8') as handle:
        prompt = _build_prompt(json.load(handle), args)
    gpu = _check_idle(
        server,
        args.max_idle_power_watts,
        args.max_idle_sm_clock_mhz,
    )
    client_id = str(uuid.uuid4())
    prompt_id, job, gpu = asyncio.run(
        _submit_and_wait(server, prompt, client_id, args)
    )
    result = {
        'prompt_id': prompt_id,
        'status': job['status'],
        'comfyui_version': version,
        'gpu_preflight': gpu,
        'execution_status': job.get('execution_status'),
        'execution_error': None,
        'outputs_count': job.get('outputs_count'),
    }
    error = job.get('execution_error') or {}
    if error:
        result['execution_error'] = {
            'node_id': error.get('node_id'),
            'node_type': error.get('node_type'),
            'exception_type': error.get('exception_type'),
            'exception_message': error.get('exception_message'),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    if job['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
