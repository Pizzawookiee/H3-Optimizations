'''CPU-only selection contract for the INT8 Triton sparse fallback.'''

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
from h3_optimizations.attention.sparse import (  # noqa: E402
    FP8FlexError,
    SparseSageError,
    TritonSparseError,
    TritonSparseSpec,
)
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    SparseRequest,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    QKV_STANDARD,
    QKV_TRITON_SPARSE_CHUNKED,
    QKVProviderResolution,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


def plan():
    return (
        H3OptimizationPlan()
        .with_memory(MemoryRequest())
        .with_sparse(SparseRequest())
    )


def inventory(convrot=True):
    return SimpleNamespace(
        qkv=(object(),),
        qkv_convrot_int8_256=bool(convrot),
        qkv_w4a8=False,
        qkv_fp8=False,
        qkv_plain_float=False,
        homogeneous=lambda name: name == 'qkv',
        labels=lambda _name: ('TensorWiseINT8Layout+convrot256',),
    )


def environment():
    return SimpleNamespace(
        cuda_available=True,
        capability=(8, 9),
        device_index=0,
    )


class TritonSparseSelectionTests(unittest.TestCase):
    def test_compatible_triton_fallback_selects_chunked_projector(self):
        with mock.patch.object(
            apply_module,
            'SPARSE_TRITON_AVAILABLE',
            True,
        ), mock.patch.object(
            apply_module,
            'preflight_triton_sparse',
            return_value=TritonSparseSpec(),
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
                plan(),
                environment(),
                inventory(),
                SparseSageError('missing sparge'),
            )

        self.assertEqual(qkv.provider_id, QKV_TRITON_SPARSE_CHUNKED)
        self.assertTrue(qkv.fused)
        self.assertEqual(attention.selected, 'triton_sparse_int8')
        self.assertEqual(attention.projector.name, 'chunked_triton_sparse_qkv')
        self.assertEqual(attention.projector.chunk_rows, 4096)
        self.assertIs(attention.backend.projector, attention.projector)

    def test_triton_is_tried_before_flex(self):
        dense_attention = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='dense',
            backend_kind='existing',
        )
        dense_qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'dense',
        )
        triton_attention = apply_module.ResolvedAttention(
            requested='sparse_sage',
            selected='triton_sparse_int8',
            backend=object(),
            reason='triton',
            backend_kind='triton_sparse_int8',
        )
        triton_qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'test',
        )

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense_attention, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=SparseSageError('missing sparge'),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            return_value=(triton_attention, triton_qkv),
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            attention, _qkv = apply_module._resolve_attention(
                plan(), object(), inventory(), environment()
            )

        self.assertEqual(attention.selected, 'triton_sparse_int8')
        triton.assert_called_once()
        flex.assert_not_called()

    def test_flex_is_used_only_after_triton_failure(self):
        dense_attention = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='dense',
            backend_kind='existing',
        )
        dense_qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'dense',
        )
        flex_attention = apply_module.ResolvedAttention(
            requested='sparse_sage',
            selected='flex_attention_fp8',
            backend=object(),
            reason='flex',
            backend_kind='flex_attention_fp8',
        )

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense_attention, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=SparseSageError('missing sparge'),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            side_effect=TritonSparseError('no triton'),
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            return_value=(flex_attention, dense_qkv),
        ) as flex:
            attention, _qkv = apply_module._resolve_attention(
                plan(), object(), inventory(), environment()
            )

        self.assertEqual(attention.selected, 'flex_attention_fp8')
        flex.assert_called_once()

    def test_dense_is_used_if_all_sparse_executors_fail(self):
        dense_attention = apply_module.ResolvedAttention(
            requested='existing',
            selected='existing',
            backend=None,
            reason='dense fallback',
            backend_kind='existing',
        )
        dense_qkv = QKVProviderResolution(
            QKV_STANDARD,
            False,
            'dense',
        )

        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(dense_attention, dense_qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            side_effect=apply_module.SparseKitchenError('native missing'),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=SparseSageError('missing sparge'),
        ), mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            side_effect=TritonSparseError('no triton'),
        ), mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            side_effect=FP8FlexError('no flex'),
        ):
            attention, qkv = apply_module._resolve_attention(
                plan(), object(), inventory(), environment()
            )

        self.assertEqual(attention.selected, 'existing')
        self.assertIs(qkv, dense_qkv)
        self.assertIn('INT8 Triton unavailable', attention.reason)
        self.assertIn('FP8 FlexAttention unavailable', attention.reason)


if __name__ == '__main__':
    unittest.main()
