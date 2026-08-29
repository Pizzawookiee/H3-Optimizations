'''CPU-only contracts for the real H3 AIMDO residency matrix.'''

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest


PACK = Path(__file__).resolve().parents[1]
BENCHMARKS = PACK / 'benchmarks'
sys.path.insert(0, str(BENCHMARKS))
SCRIPT = BENCHMARKS / 'bench_aimdo_h3_residency.py'
SPEC = importlib.util.spec_from_file_location('bench_aimdo_h3_residency', SCRIPT)
aimdo_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aimdo_bench)
bench = aimdo_bench.benchmark


class StubSchemas(bench.Schemas):
    def __init__(self):
        super().__init__('http://stub', None)
        combo = lambda values, default=None: [
            values,
            {} if default is None else {'default': default},
        ]
        self.cache = {
            'UNETLoader': {'input': {'required': {
                'unet_name': combo(['u.safetensors']),
                'weight_dtype': combo(['default']),
            }}},
            'CLIPLoader': {'input': {'required': {
                'clip_name': combo(['c.safetensors']),
                'type': combo(['minimax']),
            }}},
            'VAELoader': {'input': {'required': {
                'vae_name': combo(['v.safetensors']),
            }}},
            'MiniMaxH3ImageToVideo': {'input': {'required': {
                'clip': ['CLIP', {}],
                'vae': ['VAE', {}],
                'prompt': ['STRING', {'default': ''}],
                'width': ['INT', {'default': 1376}],
                'height': ['INT', {'default': 768}],
                'length': ['INT', {'default': 5}],
            }}},
            'H3AIMDOResidencyLimiter': {'input': {'required': {
                'model': ['MODEL', {}],
                'residency': combo(
                    ['stock', '0 blocks', '1 block', '2 blocks', '4 blocks'],
                    '2 blocks',
                ),
            }}},
            'H3BenchmarkForceQKVConfig0': {'input': {'required': {
                'model': ['MODEL', {}],
            }}},
            'BasicGuider': {'input': {'required': {
                'model': ['MODEL', {}],
                'conditioning': ['CONDITIONING', {}],
            }}},
            'KSamplerSelect': {'input': {'required': {
                'sampler_name': combo(['res_multistep']),
            }}},
            'BasicScheduler': {'input': {'required': {
                'model': ['MODEL', {}],
                'scheduler': combo(['simple']),
                'steps': ['INT', {'default': 20}],
                'denoise': ['FLOAT', {'default': 1.0}],
            }}},
            'SplitSigmas': {'input': {'required': {
                'sigmas': ['SIGMAS', {}],
                'step': ['INT', {'default': 0}],
            }}},
            'RandomNoise': {'input': {'required': {
                'noise_seed': ['INT', {'default': 0}],
            }}},
            'SamplerCustomAdvanced': {'input': {'required': {
                'noise': ['NOISE', {}],
                'guider': ['GUIDER', {}],
                'sampler': ['SAMPLER', {}],
                'sigmas': ['SIGMAS', {}],
                'latent_image': ['LATENT', {}],
            }}},
            'PreviewAny': {'input': {'required': {'source': ['*', {}]}}},
        }

    async def get(self, node_type):
        return self.cache[node_type]


def build(arm):
    args = bench.parse_args([
        '--dry-run',
        '--warmup-steps', '1',
        '--measure-steps', '1',
    ])
    return asyncio.run(bench.build_prompt(StubSchemas(), arm, 5, args))


class AIMDOH3ResidencyBenchmarkTests(unittest.TestCase):
    def test_wrapper_forces_unique_arm_seeds(self):
        self.assertEqual(
            aimdo_bench.benchmark_argv(['--dry-run']),
            ['--dry-run', '--unique-arm-seeds'],
        )

    def test_matrix_is_five_frames_with_blank_conditioning(self):
        self.assertEqual(bench.WORKLOADS, {'5f': 5})
        self.assertEqual(bench.DEFAULT_PROMPT, '')

    def test_every_arm_changes_only_the_limiter_value(self):
        expected = {
            'aimdo_stock': 'stock',
            'aimdo_0': '0 blocks',
            'aimdo_1': '1 block',
            'aimdo_2': '2 blocks',
            'aimdo_4': '4 blocks',
        }
        self.assertEqual(set(bench.ARMS), set(expected))
        for arm, residency in expected.items():
            self.assertEqual(
                bench.ARMS[arm],
                [('H3AIMDOResidencyLimiter', {'residency': residency})],
            )

    def test_graph_uses_real_h3_path_and_three_sampler_steps(self):
        graph, executed = build('aimdo_2')
        self.assertEqual(executed, 3)
        self.assertEqual(graph['cond']['inputs']['prompt'], '')
        self.assertEqual(graph['cond']['inputs']['length'], 5)
        self.assertEqual(
            graph['patch1_H3AIMDOResidencyLimiter']['inputs']['residency'],
            '2 blocks',
        )
        self.assertEqual(
            graph['guider']['inputs']['model'],
            ['patch1_H3AIMDOResidencyLimiter', 0],
        )
        self.assertEqual(graph['sample']['inputs']['latent_image'], ['cond', 1])
        self.assertEqual(graph['sink']['class_type'], 'PreviewAny')

    def test_non_patch_graph_is_identical_across_arms(self):
        graphs = {arm: build(arm)[0] for arm in bench.ARMS}
        reference = {
            key: value for key, value in graphs['aimdo_stock'].items()
            if not key.startswith('patch') and key != 'guider'
        }
        for arm, graph in graphs.items():
            current = {
                key: value for key, value in graph.items()
                if not key.startswith('patch') and key != 'guider'
            }
            self.assertEqual(current, reference, arm)


if __name__ == '__main__':
    unittest.main()
