'''Run paired exact/MLP-sharing H3 latent experiments through local ComfyUI.'''

import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

from run_mlp_sharing_experiment import (
    _check_idle,
    _json_request,
    _linked_source,
    _literal_widgets,
    _submit_and_wait,
    _widgets,
)

REQUIRED_CLASSES = (
    'UNETLoader',
    'VAELoader',
    'CLIPLoader',
    'LoadImage',
    'ZiroScaleImageToNativeCanvas',
    'ResolutionSelector',
    'MiniMaxH3ReferenceToVideoZi',
    'H3MemoryOptimization',
    'H3MLPSharing',
    'H3SparseAttentionAdvanced',
    'RandomNoise',
    'KSamplerSelect',
    'BasicScheduler',
    'BasicGuider',
    'SamplerCustomAdvancedMiniMaxLatentPair',
    'H3MLPSharingProbeOutput',
)

MATRIX_ARMS = (
    ('exact-repeat', False, 'input_cosine', '0%'),
    ('zero-path', True, 'input_cosine', '0%'),
    ('similarity-25', True, 'input_cosine', '25%'),
    ('random-25', True, 'random_local', '25%'),
    ('similarity-50', True, 'input_cosine', '50%'),
    ('random-50', True, 'random_local', '50%'),
    ('similarity-75', True, 'input_cosine', '75%'),
    ('random-75', True, 'random_local', '75%'),
    ('similarity-87_5', True, 'input_cosine', '87.5%'),
    ('random-87_5', True, 'random_local', '87.5%'),
)


def _active_node(workflow, node_type):
    return next(
        node for node in workflow['nodes']
        if node['type'] == node_type and int(node.get('mode', 0)) == 0
    )


def _build_prompt(workflow, args, arm):
    arm_name, sharing_enabled, selector, removal_fraction = arm
    ref = _active_node(workflow, 'MiniMaxH3ReferenceToVideoZi')
    clip = _linked_source(workflow, ref, 'clip')
    video_vae = _linked_source(workflow, ref, 'vae')
    audio_vae = _linked_source(workflow, ref, 'audio_vae')
    prompt_node = _linked_source(workflow, ref, 'prompt')
    resolution = _linked_source(workflow, ref, 'width')
    ref_scale = _linked_source(workflow, ref, 'ref_images.ref_image_0')
    ref_image = _linked_source(workflow, ref_scale, 'image')
    unet = _active_node(workflow, 'UNETLoader')
    memory = _active_node(workflow, 'H3MemoryOptimization')
    noise = _active_node(workflow, 'RandomNoise')
    sampler = _active_node(workflow, 'KSamplerSelect')
    scheduler = _active_node(workflow, 'BasicScheduler')
    ref_values = _widgets(ref)
    if 'value' not in _widgets(prompt_node):
        raise ValueError('the linked prompt node has no named value')

    return {
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
            'class_type': 'H3MLPSharing',
            'inputs': {
                'model': ['2', 0],
                'enabled': sharing_enabled,
                'removal_fraction': removal_fraction,
                'selector': selector,
                'start_after_step': int(args.start_after_step),
                'layers': args.layers,
                'selector_seed': int(args.selector_seed),
                'run_tag': args.run_tag + '-' + arm_name,
            },
        },
        '4': {
            'class_type': 'H3SparseAttentionAdvanced',
            'inputs': {
                'model': ['2', 0],
                'video_budget': float(args.video_budget),
                'early_steps': 0,
                'early_kv': 0.5,
                'late_steps': 0,
                'late_kv': 0.5,
            },
        },
        '5': {
            'class_type': 'H3SparseAttentionAdvanced',
            'inputs': {
                'model': ['3', 0],
                'video_budget': float(args.video_budget),
                'early_steps': 0,
                'early_kv': 0.5,
                'late_steps': 0,
                'late_kv': 0.5,
            },
        },
        '6': {
            'class_type': 'CLIPLoader',
            'inputs': _literal_widgets(clip, ('clip_name', 'type', 'device')),
        },
        '7': {
            'class_type': 'VAELoader',
            'inputs': _literal_widgets(video_vae, ('vae_name',)),
        },
        '8': {
            'class_type': 'VAELoader',
            'inputs': _literal_widgets(audio_vae, ('vae_name',)),
        },
        '9': {
            'class_type': 'LoadImage',
            'inputs': _literal_widgets(ref_image, ('image',)),
        },
        '10': {
            'class_type': 'ZiroScaleImageToNativeCanvas',
            'inputs': {
                'image': ['9', 0],
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
        '11': {
            'class_type': 'ResolutionSelector',
            'inputs': _literal_widgets(
                resolution,
                ('aspect_ratio', 'megapixels', 'multiple'),
            ),
        },
        '12': {
            'class_type': 'MiniMaxH3ReferenceToVideoZi',
            'inputs': {
                'clip': ['6', 0],
                'vae': ['7', 0],
                'audio_vae': ['8', 0],
                'prompt': _widgets(prompt_node)['value'],
                'width': ['11', 0],
                'height': ['11', 1],
                'length': int(ref_values['length']),
                'ref_image_size': ref_values['ref_image_size'],
                'cond_cache': ref_values['cond_cache'],
                'ref_images.ref_image_0': ['10', 0],
            },
        },
        '13': {
            'class_type': 'RandomNoise',
            'inputs': _literal_widgets(noise, ('noise_seed',)),
        },
        '14': {
            'class_type': 'KSamplerSelect',
            'inputs': _literal_widgets(sampler, ('sampler_name',)),
        },
        '15': {
            'class_type': 'BasicScheduler',
            'inputs': {
                'model': ['4', 0],
                **_literal_widgets(scheduler, ('scheduler', 'steps', 'denoise')),
            },
        },
        '16': {
            'class_type': 'BasicGuider',
            'inputs': {'model': ['4', 0], 'conditioning': ['12', 0]},
        },
        '17': {
            'class_type': 'BasicGuider',
            'inputs': {'model': ['5', 0], 'conditioning': ['12', 0]},
        },
        '18': {
            'class_type': 'SamplerCustomAdvancedMiniMaxLatentPair',
            'inputs': {
                'noise': ['13', 0],
                'guider_exact': ['16', 0],
                'guider_candidate': ['17', 0],
                'sampler': ['14', 0],
                'sigmas': ['15', 0],
                'latent_image': ['12', 1],
                'video_latent_frames': args.video_latent_frames,
                'pair_id': args.current_pair_id,
                'exact_label': 'exact',
                'candidate_label': arm_name,
                'vram_guard_mb': int(args.vram_guard_mb),
            },
        },
        '19': {
            'class_type': 'H3MLPSharingProbeOutput',
            'inputs': {'samples': ['18', 2]},
        },
    }


def _verify_server(server):
    stats = _json_request(server + '/api/system_stats')
    argv = stats['system'].get('argv', ())
    main = '' if not argv else str(argv[0]).replace('\\', '/').lower()
    if not main.endswith('main.py'):
        raise RuntimeError('the loopback server does not identify this Comfy checkout')
    configured = _json_request(server + '/internal/folder_paths')
    if isinstance(configured, str):
        configured = json.loads(configured)
    expected = (Path(__file__).resolve().parents[3] / 'custom_nodes').resolve()
    custom_nodes = {
        Path(value).resolve()
        for value in configured.get('custom_nodes', ())
    }
    if expected not in custom_nodes:
        raise RuntimeError('the loopback server uses a different custom-node root')
    for class_name in REQUIRED_CLASSES:
        schema = _json_request(server + '/api/object_info/' + class_name)
        if class_name not in schema:
            raise RuntimeError('server is missing required node %s' % class_name)
    return stats['system'].get('comfyui_version')


def _single_arm(args):
    return (
        args.arm_name,
        args.candidate_policy == 'sharing',
        args.selector,
        args.removal_fraction,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow', type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:8188')
    parser.add_argument('--run-tag', default='mlp-quality')
    parser.add_argument('--matrix', action='store_true')
    parser.add_argument('--arm-name', default='candidate')
    parser.add_argument('--candidate-policy', choices=('sharing', 'exact'), default='sharing')
    parser.add_argument('--selector', choices=('input_cosine', 'random_local'), default='input_cosine')
    parser.add_argument('--removal-fraction', choices=('0%', '25%', '50%', '75%', '87.5%'), default='50%')
    parser.add_argument('--start-after-step', type=int, default=3)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--selector-seed', type=int, default=0)
    parser.add_argument('--video-budget', type=float, default=0.2)
    parser.add_argument('--video-latent-frames', default='0%,20%,40%,60%,80%,100%')
    parser.add_argument(
        '--vram-guard-mb',
        type=int,
        default=0,
        help=(
            'Preview-sampler VRAM guard margin. Paired runs default to 0 because '
            'the cold H3 initialization forward has no packed latent layout.'
        ),
    )
    parser.add_argument('--max-idle-power-watts', type=float, default=60.0)
    parser.add_argument('--max-idle-sm-clock-mhz', type=float, default=500.0)
    parser.add_argument('--i-understand-this-uses-gpu', action='store_true')
    args = parser.parse_args(argv)
    if not args.i_understand_this_uses_gpu:
        parser.error('pass --i-understand-this-uses-gpu after explicit authorization')
    if args.start_after_step < 0:
        parser.error('--start-after-step must be non-negative')

    with args.workflow.open('r', encoding='utf-8') as handle:
        workflow = json.load(handle)
    server = args.server.rstrip('/')
    version = _verify_server(server)
    _check_idle(server, args.max_idle_power_watts, args.max_idle_sm_clock_mhz)
    arms = MATRIX_ARMS if args.matrix else (_single_arm(args),)
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    results = []
    for index, arm in enumerate(arms):
        args.current_pair_id = '%s-%s-%s-%02d' % (
            args.run_tag,
            stamp,
            arm[0],
            index,
        )
        prompt = _build_prompt(workflow, args, arm)
        client_id = str(uuid.uuid4())
        prompt_id, job, gpu = asyncio.run(
            _submit_and_wait(server, prompt, client_id, args)
        )
        result = {
            'arm': arm[0],
            'pair_id': args.current_pair_id,
            'prompt_id': prompt_id,
            'status': job['status'],
            'gpu_preflight': gpu,
            'capture_subdirectories': [
                'h3_latent_capture/%s-exact' % args.current_pair_id,
                'h3_latent_capture/%s-candidate' % args.current_pair_id,
            ],
            'comparison_subdirectory': (
                'h3_latent_comparison/%s' % args.current_pair_id
            ),
        }
        results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if job['status'] != 'completed':
            raise SystemExit(1)
    print(json.dumps({
        'comfyui_version': version,
        'runs': results,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
