'''CPU-only selection contract for chunked Sparse Sage QKV.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

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
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_SPARSE_CONVROT_INT8,
    QKV_TRITON_SPARSE_CHUNKED,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def sparse_spec():
    return SimpleNamespace(
        signature=('test-sparse-spec',),
        q_tile=128,
        kv_tile=64,
        qk_format='block_int8',
        q_scale_layout='per_q_tile_float32',
        k_scale_layout='per_kv_tile_float32',
        projected_v_format='floating_hnd',
        summary_format='tile_mean',
        v_format='fp16',
        accumulator='f16',
        kernel=lambda *_args: None,
    )


class SparseSelectionTests(unittest.TestCase):
    def test_compatible_sparse_path_selects_4096_row_projector(self):
        plan = (
            H3OptimizationPlan()
            .with_memory(MemoryRequest())
            .with_sparse(SparseRequest())
        )
        inventory = SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(12, 0),
        )
        spec = sparse_spec()

        with mock.patch.object(
            apply_module,
            'SPARSE_TRITON_AVAILABLE',
            True,
        ), mock.patch.object(
            apply_module,
            'preflight_sparse_sage',
            return_value=spec,
        ), mock.patch.object(
            apply_module,
            'HybridSparseBackend',
            side_effect=lambda config, **kwargs: SimpleNamespace(
                name='sparse_sage',
                config=config,
                **kwargs,
            ),
        ):
            attention, qkv = apply_module._resolve_sparse(
                plan,
                environment,
                inventory,
            )

        self.assertEqual(qkv.provider_id, QKV_SPARSE_CONVROT_INT8)
        self.assertEqual(attention.projector.name, 'chunked_sparse_sage_qkv')
        self.assertEqual(attention.projector.chunk_rows, 4096)
        self.assertIs(attention.backend.projector, attention.projector)
        self.assertIs(attention.backend.kernel_spec, spec)

    def test_triton_fallback_selects_4096_row_projector(self):
        plan = (
            H3OptimizationPlan()
            .with_memory(MemoryRequest())
            .with_sparse(SparseRequest())
        )
        inventory = SimpleNamespace(
            qkv=(object(),),
            qkv_convrot_int8_256=True,
            qkv_w4a8=False,
            qkv_fp8=False,
            qkv_plain_float=False,
            homogeneous=lambda name: name == 'qkv',
            labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
        )
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
        )
        spec = SimpleNamespace(signature=('test-triton-spec',))

        with mock.patch.object(
            apply_module,
            'preflight_triton_sparse',
            return_value=spec,
        ), mock.patch.object(
            apply_module,
            'TritonSparseBackend',
            side_effect=lambda config, **kwargs: SimpleNamespace(
                name='triton_sparse_int8',
                config=config,
                **kwargs,
            ),
        ):
            attention, qkv = apply_module._resolve_triton_sparse(
                plan,
                environment,
                inventory,
                'compiled Sparse Sage unavailable',
            )

        self.assertEqual(qkv.provider_id, QKV_TRITON_SPARSE_CHUNKED)
        self.assertEqual(attention.selected, 'triton_sparse_int8')
        self.assertEqual(attention.projector.name, 'chunked_triton_sparse_qkv')
        self.assertEqual(attention.projector.chunk_rows, 4096)
        self.assertIs(attention.backend.projector, attention.projector)
        self.assertIs(attention.backend.spec, spec)

    def test_attention_prefers_triton_before_fp8_flex(self):
        plan = H3OptimizationPlan().with_sparse(SparseRequest())
        dense = (object(), object())
        resolved = (object(), object())

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=dense,
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('Sparge missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            return_value=resolved,
        ) as triton_sparse, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as fp8_flex:
            actual = apply_module._resolve_attention(
                plan,
                object(),
                object(),
                object(),
            )

        self.assertIs(actual, resolved)
        triton_sparse.assert_called_once()
        fp8_flex.assert_not_called()


if __name__ == '__main__':
    unittest.main()
