'''CPU contracts for the synthetic non-ConvRot MLP provider benchmark.'''

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import unittest

import torch


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / 'benchmarks'
    / 'bench_nonconvrot_mlp.py'
)
SPEC = importlib.util.spec_from_file_location(
    'bench_nonconvrot_mlp',
    BENCHMARK,
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class FakeSafeOpen:
    def __init__(self, tensors):
        self.tensors = tensors

    def __call__(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, key):
        return self.tensors[key]


class FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


def quant_metadata(**changes):
    values = {
        'format': 'int8_tensorwise',
        'convrot': True,
        'transposed': False,
        'convrot_groupsize': 256,
    }
    values.update(changes)
    return torch.frombuffer(
        bytearray(json.dumps(values).encode('utf-8')),
        dtype=torch.uint8,
    ).clone()


class NonConvRotMLPBenchmarkTests(unittest.TestCase):
    def test_default_rows_cover_memory_equivalence_control(self):
        self.assertEqual(bench.DEFAULT_ROWS, (2048, 4096))
        full_2048 = bench.activation_contract(2048)[
            'full_fc1_expansion'
        ]['bytes']
        sliced_4096 = bench.activation_contract(4096)[
            'convrot_two_slice_fc1_target'
        ]['bytes']
        self.assertEqual(full_2048, 112 * 2**20)
        self.assertEqual(full_2048, sliced_4096)

    def test_exact_h3_activation_contract(self):
        contract = bench.activation_contract(4096)
        self.assertEqual(
            contract['full_fc1_expansion']['bytes'],
            224 * 2**20,
        )
        self.assertEqual(
            contract['swiglu_activation']['bytes'],
            112 * 2**20,
        )
        self.assertEqual(
            contract['convrot_two_slice_fc1_target']['bytes'],
            112 * 2**20,
        )
        self.assertEqual(
            [(item['m'], item['n'], item['k']) for item in contract['logical_gemms']],
            [(4096, 28672, 5376), (4096, 5376, 14336)],
        )
        self.assertEqual(
            [
                (item['m'], item['n'], item['k'])
                for item in contract['convrot_two_slice_gemms']
            ],
            [
                (4096, 14336, 5376),
                (4096, 5376, 7168),
                (4096, 14336, 5376),
                (4096, 5376, 7168),
            ],
        )

    def test_rows_are_aligned_and_unique(self):
        self.assertEqual(bench.parse_rows('2048,4096'), (2048, 4096))
        for value in ('', '1', '2048,2048', '-256'):
            with self.assertRaises(Exception):
                bench.parse_rows(value)

    def test_exact_h3_shapes_fail_closed(self):
        self.assertEqual(
            bench.validate_h3_shapes((28672, 5376), (5376, 14336)),
            (5376, 14336, 28672),
        )
        with self.assertRaisesRegex(ValueError, 'exact H3 shape'):
            bench.validate_h3_shapes((28672, 4096), (4096, 14336))

    def test_convrot_wrapper_preserves_checkpoint_contract(self):
        class Params:
            def __init__(self, **values):
                self.__dict__.update(values)

        class Layout:
            pass

        class Quantized:
            def __init__(self, qdata, layout, params):
                self.qdata = qdata
                self.layout = layout
                self.params = params

        Layout.Params = Params
        qdata = torch.zeros((4, 2), dtype=torch.int8)
        wrapped = bench._convrot_weight(
            torch,
            Quantized,
            Layout,
            {'weight': qdata, 'scale': torch.ones((4, 1))},
        )
        self.assertIs(wrapped.qdata, qdata)
        self.assertEqual(wrapped.layout, 'TensorWiseINT8Layout')
        self.assertEqual(wrapped.params.orig_shape, (4, 2))
        self.assertEqual(wrapped.params.orig_dtype, torch.bfloat16)
        self.assertTrue(wrapped.params.convrot)
        self.assertEqual(wrapped.params.convrot_groupsize, 256)

    def test_checkpoint_loader_requires_complete_convrot_pair(self):
        prefix = 'model.diffusion_model.blocks.0.mlp.'
        tensors = {
            prefix + 'fc1.weight': FakeTensor(
                (bench.EXPANDED, bench.HIDDEN),
                torch.int8,
            ),
            prefix + 'fc1.weight_scale': FakeTensor(
                (bench.EXPANDED,),
                torch.float32,
            ),
            prefix + 'fc1.comfy_quant': quant_metadata(),
            prefix + 'fc2.weight': FakeTensor(
                (bench.HIDDEN, bench.FFN),
                torch.int8,
            ),
            prefix + 'fc2.weight_scale': FakeTensor(
                (bench.HIDDEN,),
                torch.float32,
            ),
            prefix + 'fc2.comfy_quant': quant_metadata(),
        }
        loaded = bench.load_convrot_mlp(
            'model.safetensors',
            safe_open_fn=FakeSafeOpen(tensors),
        )
        self.assertEqual(loaded['prefix'], prefix)
        self.assertEqual(loaded['fc1']['weight'].shape, (28672, 5376))
        self.assertEqual(loaded['fc2']['weight'].shape, (5376, 14336))

        tensors[prefix + 'fc2.comfy_quant'] = quant_metadata(convrot=False)
        with self.assertRaisesRegex(ValueError, 'ConvRot-256'):
            bench.load_convrot_mlp(
                'model.safetensors',
                safe_open_fn=FakeSafeOpen(tensors),
            )

    def test_cli_requires_explicit_gpu_acknowledgement(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args(['--checkpoint', 'model.safetensors'])
        args = bench.parse_args(
            [
                '--checkpoint',
                'model.safetensors',
                '--rows',
                '2048,4096',
                '--i-understand-this-uses-gpu',
            ]
        )
        self.assertEqual(args.rows, (2048, 4096))


if __name__ == '__main__':
    unittest.main()
