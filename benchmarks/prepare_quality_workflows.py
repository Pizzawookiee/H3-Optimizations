'''Prepare matched UI workflows for the dense Kitchen quality comparison.'''

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


FULL_NAME = 'H3_Kitchen_Full_QKV.json'
CHUNKED_NAME = 'H3_Kitchen_Chunked_QKV.json'
MANIFEST_NAME = 'H3_Kitchen_QKV_Quality.json'


def _single_node(workflow, node_type):
    matches = [node for node in workflow['nodes'] if node.get('type') == node_type]
    if len(matches) != 1:
        raise ValueError(
            'expected exactly one %s node, found %d' % (node_type, len(matches))
        )
    return matches[0]


def _inputs_by_name(node):
    return {item['name']: item for item in node.get('inputs', ())}


def _normalize_h3_nodes(workflow, fused_qkv):
    memory = _single_node(workflow, 'H3MemoryOptimization')
    memory_inputs = _inputs_by_name(memory)
    required_memory = ('model', 'fused_qkv', 'mlp_memory', 'chunk_rows')
    missing = [name for name in required_memory if name not in memory_inputs]
    if missing:
        raise ValueError('H3MemoryOptimization is missing inputs: %s' % ', '.join(missing))
    memory['inputs'] = [memory_inputs[name] for name in required_memory]
    memory['widgets_values'] = [fused_qkv, 'auto', 2048]
    memory['mode'] = 0

    sparse = _single_node(workflow, 'H3SparseAttention')
    sparse_inputs = _inputs_by_name(sparse)
    required_sparse = ('model', 'video_budget', 'denser_early_late_steps')
    missing = [name for name in required_sparse if name not in sparse_inputs]
    if missing:
        raise ValueError('H3SparseAttention is missing inputs: %s' % ', '.join(missing))
    sparse['inputs'] = [sparse_inputs[name] for name in required_sparse]
    sparse['widgets_values'] = [0.5, False]
    sparse['mode'] = 4


def _set_output_prefix(workflow, prefix):
    save = _single_node(workflow, 'SaveVideo')
    values = list(save.get('widgets_values', ()))
    if not values:
        raise ValueError('SaveVideo has no filename prefix widget')
    values[0] = prefix
    save['widgets_values'] = values


def _set_reference_image(workflow, filename):
    connected = []
    for node in workflow['nodes']:
        if node.get('type') != 'LoadImage':
            continue
        if any(output.get('links') for output in node.get('outputs', ())):
            connected.append(node)
    if len(connected) != 1:
        raise ValueError(
            'expected exactly one connected LoadImage node, found %d'
            % len(connected)
        )
    values = list(connected[0].get('widgets_values', ()))
    if not values:
        raise ValueError('connected LoadImage has no image widget')
    values[0] = filename
    connected[0]['widgets_values'] = values


def prepare_workflow(source, fused_qkv, prefix, reference_image=None):
    if fused_qkv not in ('auto', 'off'):
        raise ValueError('fused_qkv must be auto or off')
    workflow = deepcopy(source)
    _normalize_h3_nodes(workflow, fused_qkv)
    _set_output_prefix(workflow, prefix)
    if reference_image is not None:
        _set_reference_image(workflow, reference_image)
    return workflow


def _widgets(workflow, node_type):
    return list(_single_node(workflow, node_type).get('widgets_values', ()))


def _resolution(workflow):
    aspect, megapixels, multiple = _widgets(workflow, 'ResolutionSelector')
    ratios = {
        '1:1 (Square)': (1, 1),
        '2:3 (Portrait Photo)': (2, 3),
        '3:2 (Photo)': (3, 2),
        '3:4 (Portrait Standard)': (3, 4),
        '4:3 (Standard)': (4, 3),
        '9:16 (Portrait Widescreen)': (9, 16),
        '16:9 (Widescreen)': (16, 9),
        '21:9 (Ultrawide)': (21, 9),
    }
    if aspect not in ratios:
        raise ValueError('unknown ResolutionSelector aspect ratio %r' % aspect)
    width_ratio, height_ratio = ratios[aspect]
    scale = math.sqrt(float(megapixels) * 1024 * 1024 / (width_ratio * height_ratio))
    return {
        'aspect_ratio': aspect,
        'megapixels': float(megapixels),
        'multiple': int(multiple),
        'width': round(width_ratio * scale / int(multiple)) * int(multiple),
        'height': round(height_ratio * scale / int(multiple)) * int(multiple),
    }


def comparison_manifest(source_bytes, full, chunked, source_path):
    full_memory = _widgets(full, 'H3MemoryOptimization')
    chunked_memory = _widgets(chunked, 'H3MemoryOptimization')
    if full_memory[1:] != chunked_memory[1:]:
        raise ValueError('dense workflows differ outside fused_qkv')
    noise = _widgets(full, 'RandomNoise')
    if len(noise) < 2 or noise[1] != 'fixed':
        raise ValueError('RandomNoise must use a fixed seed')
    scheduler = _widgets(full, 'BasicScheduler')
    duration = _widgets(full, 'PrimitiveFloat')
    return {
        'source': str(source_path.resolve()),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'cases': {
            'full_kitchen': {
                'workflow': FULL_NAME,
                'fused_qkv': 'off',
                'output_prefix': _widgets(full, 'SaveVideo')[0],
            },
            'chunked_kitchen': {
                'workflow': CHUNKED_NAME,
                'fused_qkv': 'auto',
                'output_prefix': _widgets(chunked, 'SaveVideo')[0],
            },
        },
        'shared': {
            'diffusion_model': _widgets(full, 'UNETLoader')[0],
            'text_encoder': _widgets(full, 'CLIPLoader')[0],
            'vae_models': [values[0] for values in (
                list(node.get('widgets_values', ()))
                for node in full['nodes']
                if node.get('type') == 'VAELoader'
            )],
            'noise_seed': int(noise[0]),
            'scheduler': scheduler[0],
            'steps': int(scheduler[1]),
            'denoise': float(scheduler[2]),
            'duration_seconds': float(duration[0]),
            'resolution': _resolution(full),
            'mlp_memory': full_memory[1],
            'mlp_chunk_rows': int(full_memory[2]),
            'sparse_attention': 'bypassed',
        },
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n',
        encoding='utf-8',
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Prepare matched full and chunked Kitchen UI workflows.'
    )
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-image')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes.decode('utf-8-sig'))
    full = prepare_workflow(
        source,
        'off',
        'video/H3_Kitchen_Full_QKV',
        args.reference_image,
    )
    chunked = prepare_workflow(
        source,
        'auto',
        'video/H3_Kitchen_Chunked_QKV',
        args.reference_image,
    )
    manifest = comparison_manifest(source_bytes, full, chunked, args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / FULL_NAME, full)
    _write_json(args.output_dir / CHUNKED_NAME, chunked)
    _write_json(args.output_dir / MANIFEST_NAME, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
