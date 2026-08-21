'''CPU tests for the fixed dense Kitchen workflow preparation helper.'''

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import unittest


PACK = Path(__file__).resolve().parents[1]
SCRIPT = PACK / 'benchmarks' / 'prepare_quality_workflows.py'
SPEC = importlib.util.spec_from_file_location('prepare_quality_workflows', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def node(node_id, node_type, inputs, widgets, outputs=None):
    return {
        'id': node_id,
        'type': node_type,
        'mode': 0,
        'inputs': [
            {'name': name, 'link': node_id if name == 'model' else None}
            for name in inputs
        ],
        'outputs': outputs or [],
        'widgets_values': widgets,
    }


def workflow():
    return {
        'nodes': [
            node(1, 'H3MemoryOptimization', (
                'model', 'enabled', 'attention', 'fused_qkv', 'mlp_memory',
                'chunk_rows', 'prefer_held_weights',
            ), (True, 'auto', 'auto', 'auto', 2048, True), [
                {'name': 'MODEL', 'links': [10]},
            ]),
            node(2, 'H3SparseAttention', (
                'model', 'enabled', 'video_budget', 'denser_early_late_steps',
            ), (True, 0.1, True)),
            node(3, 'SaveVideo', ('video',), ('old', 'auto', 'auto')),
            node(4, 'LoadImage', ('image',), ('unused.png',), [
                {'name': 'IMAGE', 'links': []},
            ]),
            node(5, 'LoadImage', ('image',), ('missing.png',), [
                {'name': 'IMAGE', 'links': [11]},
            ]),
            node(6, 'RandomNoise', ('noise_seed',), (1234, 'fixed')),
            node(7, 'BasicScheduler', ('model',), ('beta', 20, 1.0)),
            node(8, 'ResolutionSelector', (), ('4:3 (Standard)', 1.0, 32)),
            node(9, 'PrimitiveFloat', (), (10.0,)),
            node(10, 'UNETLoader', (), ('model.safetensors', 'default')),
            node(11, 'CLIPLoader', (), ('clip.safetensors', 'minimax', 'default')),
            node(12, 'VAELoader', (), ('video_vae.safetensors',)),
            node(13, 'VAELoader', (), ('audio_vae.safetensors',)),
        ],
    }


class PrepareQualityWorkflowTests(unittest.TestCase):
    def test_normalizes_retired_widgets_and_changes_only_qkv_case(self):
        source = workflow()
        full = MODULE.prepare_workflow(
            source, 'off', 'video/full', 'reference.png'
        )
        chunked = MODULE.prepare_workflow(
            source, 'auto', 'video/chunked', 'reference.png'
        )

        full_memory = MODULE._single_node(full, 'H3MemoryOptimization')
        chunked_memory = MODULE._single_node(chunked, 'H3MemoryOptimization')
        self.assertEqual(
            [item['name'] for item in full_memory['inputs']],
            ['model', 'fused_qkv', 'mlp_memory', 'chunk_rows'],
        )
        self.assertEqual(full_memory['widgets_values'], ['off', 'auto', 2048])
        self.assertEqual(chunked_memory['widgets_values'], ['auto', 'auto', 2048])
        self.assertEqual(MODULE._single_node(full, 'H3SparseAttention')['mode'], 4)
        self.assertEqual(
            MODULE._single_node(full, 'H3SparseAttention')['widgets_values'],
            [0.3, False],
        )
        connected = [
            item for item in full['nodes']
            if item['type'] == 'LoadImage' and any(
                output.get('links') for output in item['outputs']
            )
        ]
        self.assertEqual(connected[0]['widgets_values'][0], 'reference.png')
        self.assertEqual(MODULE._widgets(full, 'SaveVideo')[0], 'video/full')
        self.assertEqual(MODULE._widgets(chunked, 'SaveVideo')[0], 'video/chunked')

        normalized_full = deepcopy(full)
        normalized_chunked = deepcopy(chunked)
        MODULE._single_node(normalized_full, 'H3MemoryOptimization')[
            'widgets_values'
        ][0] = 'case'
        MODULE._single_node(normalized_chunked, 'H3MemoryOptimization')[
            'widgets_values'
        ][0] = 'case'
        MODULE._single_node(normalized_full, 'SaveVideo')['widgets_values'][0] = 'case'
        MODULE._single_node(normalized_chunked, 'SaveVideo')['widgets_values'][0] = 'case'
        self.assertEqual(normalized_full, normalized_chunked)

    def test_manifest_records_fixed_contract(self):
        source = workflow()
        source_bytes = json.dumps(source).encode()
        full = MODULE.prepare_workflow(source, 'off', 'video/full')
        chunked = MODULE.prepare_workflow(source, 'auto', 'video/chunked')
        manifest = MODULE.comparison_manifest(
            source_bytes, full, chunked, Path('source.json')
        )

        self.assertEqual(
            manifest['source_sha256'], hashlib.sha256(source_bytes).hexdigest()
        )
        self.assertEqual(manifest['shared']['noise_seed'], 1234)
        self.assertEqual(manifest['shared']['steps'], 20)
        self.assertEqual(
            manifest['shared']['resolution']['width'],
            1184,
        )
        self.assertEqual(
            manifest['shared']['resolution']['height'],
            896,
        )
        self.assertEqual(manifest['shared']['sparse_attention'], 'bypassed')

    def test_rejects_non_fixed_noise(self):
        source = workflow()
        noise = MODULE._single_node(source, 'RandomNoise')
        noise['widgets_values'] = [1234, 'randomize']
        full = MODULE.prepare_workflow(source, 'off', 'video/full')
        chunked = MODULE.prepare_workflow(source, 'auto', 'video/chunked')
        with self.assertRaisesRegex(ValueError, 'fixed seed'):
            MODULE.comparison_manifest(
                b'{}', full, chunked, Path('source.json')
            )


if __name__ == '__main__':
    unittest.main()
