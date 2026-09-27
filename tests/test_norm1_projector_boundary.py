'''CPU contracts for the private lazy-Norm1 QKV handoff.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import Attention  # noqa: E402

from h3_optimizations.attention_forward import make_forward  # noqa: E402
from h3_optimizations.normalized_rows import (  # noqa: E402
    NORM1_SOURCE_KEY,
    NormalizedRows,
    NormalizedRowsUnsupported,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class _DenseEchoBackend:
    name = 'test_dense_echo'

    def __init__(self):
        self.options = []

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        del layer_index
        self.options.append(dict(transformer_options))
        return q, k, v

    @staticmethod
    def requires_fallback_inputs(_prepared):
        return False

    @staticmethod
    def execute(prepared):
        q, _k, _v = prepared
        return q


class _DecliningProjector:
    name = 'test_declining_projector'

    def __init__(self):
        self.inputs = []
        self.options = []

    def try_project(
        self,
        _module,
        x,
        _rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        del layer_index
        self.inputs.append(x)
        self.options.append(dict(transformer_options))
        return None


class _TensorOnlyProjector(_DecliningProjector):
    name = 'test_tensor_only_projector'

    def try_project(self, module, x, rope_freqs, **kwargs):
        self.inputs.append(x)
        self.options.append(dict(kwargs['transformer_options']))
        if not torch.is_tensor(x):
            raise NormalizedRowsUnsupported('projector needs a concrete tensor')
        return None


class _ProjectedProjector(_DecliningProjector):
    def try_project(self, module, x, rope_freqs, **kwargs):
        del module, rope_freqs
        self.inputs.append(x)
        self.options.append(dict(kwargs['transformer_options']))
        return object()


class _ProjectedBackend:
    name = 'test_projected'

    def __init__(self):
        self.execution_module = None

    @staticmethod
    def prepare_projected(projected, **_kwargs):
        return projected

    def execute_projected(self, module, _prepared):
        self.execution_module = module
        return torch.zeros(5, 32)


class Norm1ProjectorBoundaryTests(unittest.TestCase):
    @staticmethod
    def _module():
        module = Attention(
            hidden=32,
            heads=2,
            head_dim=16,
            eps=1e-6,
            dtype=torch.float32,
            device='cpu',
            operations=comfy.ops.disable_weight_init,
        )
        torch.manual_seed(411)
        for parameter in module.parameters():
            parameter.detach().copy_(torch.randn_like(parameter) * 0.03)
            parameter.requires_grad_(False)
        return module

    @staticmethod
    def _source(x):
        def norm(rows):
            return rows.clone()

        def modulate(rows, _shift, _scale, _selector):
            rows.add_(0.125)
            return rows

        return NormalizedRows(
            x,
            norm,
            ((0, int(x.shape[0]), 0),),
            None,
            None,
            modulate,
        )

    def test_projector_gets_lazy_source_but_backend_does_not(self):
        module = self._module()
        backend = _DenseEchoBackend()
        projector = _DecliningProjector()
        forward = make_forward(
            module,
            3,
            backend=backend,
            projector=projector,
        )
        self.assertTrue(forward._h3_optimizations_lazy_norm_source)

        torch.manual_seed(412)
        residual = torch.randn(7, 32)
        source = self._source(residual)
        actual = forward(
            residual,
            transformer_options={
                'kept': 'yes',
                NORM1_SOURCE_KEY: source,
            },
        )

        self.assertIs(projector.inputs[0], source)
        self.assertNotIn(NORM1_SOURCE_KEY, projector.options[0])
        self.assertNotIn(NORM1_SOURCE_KEY, backend.options[0])
        self.assertEqual(backend.options[0]['kept'], 'yes')

        reference_backend = _DenseEchoBackend()
        reference_projector = _DecliningProjector()
        reference = make_forward(
            module,
            3,
            backend=reference_backend,
            projector=reference_projector,
        )(
            source.materialize(),
            transformer_options={'kept': 'yes'},
        )
        torch.testing.assert_close(actual, reference)

    def test_tensor_only_projector_retries_after_local_materialization(self):
        module = self._module()
        backend = _DenseEchoBackend()
        projector = _TensorOnlyProjector()
        forward = make_forward(
            module,
            4,
            backend=backend,
            projector=projector,
        )

        residual = torch.randn(5, 32)
        source = self._source(residual)
        forward(
            residual,
            transformer_options={NORM1_SOURCE_KEY: source},
        )

        self.assertEqual(len(projector.inputs), 2)
        self.assertIs(projector.inputs[0], source)
        self.assertTrue(torch.is_tensor(projector.inputs[1]))
        self.assertNotIn(NORM1_SOURCE_KEY, projector.options[0])
        self.assertNotIn(NORM1_SOURCE_KEY, projector.options[1])

    def test_native_convrot_out_projection_exposes_residual_method_to_backend(self):
        module = self._module()
        backend = _ProjectedBackend()
        projector = _ProjectedProjector()
        source = self._source(torch.randn(5, 32))
        source.enable_attention_residual(torch.ones(1, 32))
        projection = SimpleNamespace(
            linear=lambda value: value,
            residual_=lambda *_args: True,
            release=mock.Mock(),
        )

        with mock.patch(
            'h3_optimizations.qkv.formats.describe_linear',
            return_value=SimpleNamespace(convrot_int8_256=True),
        ), mock.patch(
            'h3_optimizations.qkv.int8.LazyConvRotINT8Linear',
            return_value=projection,
        ) as lazy:
            actual = make_forward(
                module,
                5,
                backend=backend,
                projector=projector,
            )(
                source._x,
                transformer_options={NORM1_SOURCE_KEY: source},
            )

        self.assertEqual(tuple(actual.shape), (5, 32))
        lazy.assert_called_once_with(
            module.out_proj,
            allow_float_conversion=False,
        )
        self.assertIs(
            backend.execution_module._h3_out_proj_residual,
            projection.residual_,
        )
        projection.release.assert_called_once_with()


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *TEST_ARGS])
