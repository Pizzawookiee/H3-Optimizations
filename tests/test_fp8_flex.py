'''CPU-only contracts for the FP8 FlexAttention sparse fallback.'''

from copy import deepcopy
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

import h3_optimizations.apply as apply_module  # noqa: E402
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
)
from h3_optimizations.attention.sparse.fp8_flex import (  # noqa: E402
    FP8FlexBackend,
    FP8FlexError,
    FP8FlexSpec,
    block_mask_from_delta_lut,
    load_fp8_flex_spec,
    preflight_fp8_flex,
)
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    SparseRequest,
    STATUS_KEY,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKVProviderResolution,
    QKV_STANDARD,
)
from h3_optimizations.status import format_sparse_status  # noqa: E402
from torch.nn.attention.flex_attention import BlockMask  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeModel:
    def __init__(self, options=None):
        self.model_options = deepcopy(options or {})
        self.object_patches = {}

    def clone(self):
        cloned = FakeModel(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        return cloned


class FakeMetadata:
    def as_dict(self):
        return {'requested_video_budget': 0.5}


class FakeRouter:
    q_tile = 128
    kv_tile = 64

    def build_lut(self, q, _k, _layout, _video_budget):
        q_tiles = (q.shape[-2] + self.q_tile - 1) // self.q_tile
        kv_tiles = (q.shape[-2] + self.kv_tile - 1) // self.kv_tile
        dense_delta = torch.ones(kv_tiles, dtype=torch.int32)
        dense_delta[0] = 0
        lut = dense_delta.view(1, 1, 1, -1).expand(
            q.shape[0],
            q.shape[1],
            q_tiles,
            -1,
        ).clone()
        valid = torch.full(
            (q.shape[0], q.shape[1], q_tiles),
            kv_tiles,
            dtype=torch.int32,
        )
        return lut, valid, FakeMetadata()


class FP8FlexTests(unittest.TestCase):
    @staticmethod
    def _spec(attention=lambda *_args, **_kwargs: None):
        return FP8FlexSpec(
            version='test-flex',
            attention=attention,
            block_mask_type=BlockMask,
        )

    def test_preflight_requires_cuda_fp8_and_dynamo(self):
        with self.assertRaisesRegex(FP8FlexError, 'requires NVIDIA CUDA'):
            preflight_fp8_flex(
                cuda_available=lambda: False,
                capability_getter=lambda: None,
            )

        with self.assertRaisesRegex(FP8FlexError, 'unsupported'):
            preflight_fp8_flex(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 0),
                fp8_supported=lambda: False,
            )

        with self.assertRaisesRegex(FP8FlexError, 'Dynamo'):
            preflight_fp8_flex(
                cuda_available=lambda: True,
                capability_getter=lambda: (8, 9),
                fp8_supported=lambda: True,
                dynamo_supported=lambda: False,
            )

        spec = self._spec()
        self.assertIs(
            preflight_fp8_flex(
                cuda_available=lambda: True,
                capability_getter=lambda: (12, 0),
                fp8_supported=lambda: True,
                dynamo_supported=lambda: True,
                loader=lambda: spec,
            ),
            spec,
        )

    def test_loader_compiles_flex_attention_for_sparse_execution(self):
        compiled_attention = object()
        with mock.patch.object(
            torch,
            'compile',
            return_value=compiled_attention,
        ) as compile_attention:
            spec = load_fp8_flex_spec()

        self.assertIs(spec.attention, compiled_attention)
        compile_attention.assert_called_once_with(
            mock.ANY,
            fullgraph=True,
        )

    def test_delta_lut_becomes_compact_flex_block_indices(self):
        lut = torch.tensor(
            [[[[0, 2, 1], [0, 1, 1]]]],
            dtype=torch.int32,
        )
        valid = torch.tensor([[[2, 3]]], dtype=torch.int32)

        block_mask = block_mask_from_delta_lut(
            self._spec(),
            lut,
            valid,
            192,
        )

        self.assertEqual(block_mask.BLOCK_SIZE, (128, 64))
        self.assertEqual(block_mask.seq_lengths, (192, 192))
        self.assertTrue(torch.equal(block_mask.kv_num_blocks, valid))
        self.assertEqual(
            block_mask.kv_indices.tolist(),
            [[[[0, 2, 3], [0, 1, 2]]]],
        )
        self.assertIsNone(block_mask.q_indices)

    def test_backend_keeps_q_floating_and_quantizes_scaled_kv(self):
        calls = []

        def attention(q, k, v, **kwargs):
            calls.append((q, k, v, kwargs))
            return torch.ones_like(q)

        backend = FP8FlexBackend(
            HybridSparseConfig(video_budget=0.5),
            spec=self._spec(attention),
            router=FakeRouter(),
            chunk_rows=128,
            allow_cpu_for_tests=True,
        )
        q = torch.randn((1, 2, 192, 128), dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        k[:, 0].fill_(2)
        k[:, 1].fill_(4)
        v[:, 0].fill_(3)
        v[:, 1].fill_(6)
        snapshot = SimpleNamespace(
            valid_layout=True,
            error=None,
            layout=SimpleNamespace(seq_len=192),
            step_index=4,
            total_steps=20,
        )

        with mock.patch.object(
            backend,
            '_snapshot',
            return_value=snapshot,
        ):
            prepared = backend.prepare(
                q,
                k,
                v,
                layer_index=7,
                transformer_options={},
            )

        self.assertEqual(prepared.q.dtype, torch.bfloat16)
        self.assertNotEqual(prepared.q.data_ptr(), q.data_ptr())
        self.assertEqual(prepared.k_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(prepared.v_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(prepared.k_fp8.stride(-1), 1)
        self.assertEqual(prepared.v_fp8.stride(-2), 1)
        torch.testing.assert_close(
            prepared.k_scale,
            torch.tensor([[2 / 448, 4 / 448]], dtype=torch.float32),
        )
        torch.testing.assert_close(
            prepared.v_scale,
            torch.tensor([[3 / 448, 6 / 448]], dtype=torch.float32),
        )

        output = backend.execute(prepared)

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertEqual(len(calls), 1)
        call_q, call_k, call_v, kwargs = calls[0]
        self.assertIs(call_q, prepared.q)
        self.assertIs(call_k, prepared.k_fp8)
        self.assertIs(call_v, prepared.v_fp8)
        self.assertIs(kwargs['block_mask'], prepared.block_mask)
        self.assertEqual(kwargs['kernel_options']['BLOCK_M'], 128)
        self.assertEqual(kwargs['kernel_options']['BLOCK_N'], 64)
        self.assertTrue(
            kwargs['kernel_options']['ROWS_GUARANTEED_SAFE']
        )
        self.assertIs(
            kwargs['score_mod'].__closure__[0].cell_contents,
            prepared.k_scale,
        )
        restored = kwargs['score_mod'](
            torch.tensor(2.0),
            torch.tensor(0),
            torch.tensor(1),
            torch.tensor(0),
            torch.tensor(0),
        )
        torch.testing.assert_close(restored, prepared.k_scale[0, 1] * 2)
        torch.testing.assert_close(
            output[:, 0].float(),
            torch.full_like(output[:, 0].float(), 3 / 448),
            atol=2e-5,
            rtol=2e-3,
        )

    def test_selection_prefers_flex_then_keeps_dense_as_final_fallback(self):
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        dense_qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'standard projection',
        )
        dense = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='normal Comfy attention',
            backend_kind='existing',
            dense_resolution=SimpleNamespace(backend=None),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(12, 0),
            device_index=0,
        )

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('ABI missing'),
        ), mock.patch.object(
            apply_module,
            'preflight_fp8_flex',
            return_value=self._spec(),
        ):
            attention, qkv = apply_module._resolve_attention(
                plan,
                object(),
                object(),
                environment,
            )

        self.assertEqual(attention.selected, 'flex_attention_fp8')
        self.assertEqual(attention.requested, 'sparse_sage')
        self.assertEqual(qkv.provider_id, QKV_STANDARD)
        self.assertIn('ABI missing', attention.reason)

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('ABI missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            side_effect=apply_module.FP8FlexError('FP8 missing'),
        ):
            attention, qkv = apply_module._resolve_attention(
                plan,
                object(),
                object(),
                environment,
            )

        self.assertEqual(attention.selected, 'existing')
        self.assertIs(qkv, dense_qkv)
        self.assertIn('FP8 missing', attention.reason)

    def test_sparse_sage_success_does_not_probe_flex(self):
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        resolved = (
            apply_module.ResolvedAttention(
                requested='sparse_sage',
                selected='sparse_sage',
                backend=object(),
                reason='Sparse Sage selected',
                backend_kind='sparse_sage',
            ),
            QKVProviderResolution(
                QKV_STANDARD,
                False,
                'standard projection',
            ),
        )

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=resolved,
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            return_value=resolved,
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            actual = apply_module._resolve_attention(
                plan,
                object(),
                object(),
                object(),
            )

        self.assertIs(actual, resolved)
        flex.assert_not_called()

    def test_apply_installs_flex_as_the_sparse_execution_backend(self):
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'standard projection',
        )
        mlp = MLPProviderResolution('off', 'off', 'disabled')
        backend = SimpleNamespace(name='flex_attention_fp8')
        attention = apply_module.ResolvedAttention(
            requested='sparse_sage',
            selected='flex_attention_fp8',
            backend=backend,
            reason='Sparse Sage unavailable; using FP8 FlexAttention',
            backend_kind='flex_attention_fp8',
        )
        environment = SimpleNamespace(
            cuda_available=True,
            device_index=0,
            capability=(12, 0),
            device_name='fake NVIDIA',
            backend='nvidia_cuda',
            architecture='sm120',
        )
        inventory = SimpleNamespace(labels=lambda _name: ())

        with mock.patch.object(
            apply_module,
            'is_minimax_h3',
            return_value=True,
        ), mock.patch.object(
            apply_module,
            'get_h3_blocks',
            return_value=(object(),),
        ), mock.patch.object(
            apply_module,
            'inspect_h3_linears',
            return_value=inventory,
        ), mock.patch.object(
            apply_module.RuntimeEnvironment,
            'detect',
            return_value=environment,
        ), mock.patch.object(
            apply_module,
            '_resolve_attention',
            return_value=(attention, qkv),
        ), mock.patch.object(
            apply_module,
            'configure_backend',
            return_value=(backend, 50),
        ) as configure, mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 0),
        ), mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
            return_value=(object(), True),
        ) as runtime:
            patched = apply_module.apply_plan(FakeModel(), plan)

        configure.assert_called_once_with(
            mock.ANY,
            backend,
            projector=None,
        )
        runtime.assert_called_once()
        status = patched.model_options['transformer_options'][STATUS_KEY]
        self.assertEqual(status['attention']['selected'], 'flex_attention_fp8')
        self.assertTrue(status['runtime_installed'])
        self.assertIn('Attention: FP8 FlexAttention', format_sparse_status(patched))


if __name__ == '__main__':
    unittest.main()
