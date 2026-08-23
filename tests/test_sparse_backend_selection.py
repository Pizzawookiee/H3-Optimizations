'''CPU contracts for explicit sparse backend selection.'''

from copy import deepcopy
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
    PLAN_KEY,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_FLEX,
    SPARSE_BACKEND_SAGE,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_TRITON,
    STATUS_KEY,
    SparseRequest,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKVProviderResolution,
)
from h3_optimizations.status import format_sparse_status  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeModel:
    def __init__(self, options=None):
        self.model_options = deepcopy(options or {})
        self.object_patches = {}

    def clone(self):
        cloned = FakeModel(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        return cloned


def qkv_resolution():
    return QKVProviderResolution(
        'standard_h3_qkv',
        False,
        'synthetic',
    )


def resolved(kind, dense_resolution=None):
    return apply_module.ResolvedAttention(
        requested=apply_module.ATTENTION_SPARSE,
        selected=kind,
        backend=SimpleNamespace(name=kind),
        reason='synthetic',
        backend_kind=kind,
        projector=None,
        dense_resolution=dense_resolution,
    )


class SparseBackendSelectionTests(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.inventory = object()
        self.environment = object()
        self.qkv = qkv_resolution()

    def test_forced_sparse_sage_does_not_fall_through(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_SAGE)
        )
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=apply_module.SparseSageError('synthetic unavailable'),
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            with self.assertRaisesRegex(
                apply_module.SparseSageError,
                'synthetic unavailable',
            ):
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                )
        dense.assert_not_called()
        sage.assert_called_once_with(plan, self.environment, self.inventory)
        triton.assert_not_called()
        flex.assert_not_called()

    def test_forced_triton_bypasses_sage_flex_and_dense(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_TRITON)
        )
        target = (resolved(apply_module.ATTENTION_TRITON_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
            return_value=target,
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
            None,
        )
        flex.assert_not_called()

    def test_forced_kitchen_bypasses_every_other_backend(self):
        """Kitchen INT8 is explicit-only and must not reach the auto chain."""
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_KITCHEN)
        )
        target = (resolved(apply_module.ATTENTION_KITCHEN_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
        ) as flex, mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
            return_value=target,
        ) as kitchen:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_not_called()
        flex.assert_not_called()
        kitchen.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
        )

    def test_auto_never_selects_kitchen(self):
        """auto stays Sage -> Triton -> Flex -> dense until the A/B has run."""
        plan = H3OptimizationPlan(sparse=SparseRequest())
        target = (resolved(apply_module.ATTENTION_SPARSE), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
            return_value=(resolved('dense'), self.qkv),
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            return_value=target,
        ), mock.patch.object(
            apply_module,
            '_resolve_kitchen_sparse',
        ) as kitchen:
            apply_module._resolve_attention(
                plan,
                self.model,
                self.inventory,
                self.environment,
            )
        kitchen.assert_not_called()

    def test_forced_flex_bypasses_sage_triton_and_dense(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(backend=SPARSE_BACKEND_FLEX)
        )
        target = (resolved(apply_module.ATTENTION_FP8_FLEX), self.qkv)
        with mock.patch.object(
            apply_module,
            '_resolve_dense',
        ) as dense, mock.patch.object(
            apply_module,
            '_resolve_sparse',
        ) as sage, mock.patch.object(
            apply_module,
            '_resolve_triton_sparse',
        ) as triton, mock.patch.object(
            apply_module,
            '_resolve_fp8_flex',
            return_value=target,
        ) as flex:
            self.assertIs(
                apply_module._resolve_attention(
                    plan,
                    self.model,
                    self.inventory,
                    self.environment,
                ),
                target,
            )
        dense.assert_not_called()
        sage.assert_not_called()
        triton.assert_not_called()
        flex.assert_called_once_with(
            plan,
            self.environment,
            self.inventory,
            None,
            None,
        )

    def test_only_auto_enables_flex_dense_runtime_fallback(self):
        inventory = SimpleNamespace(labels=lambda _name: ())
        environment = SimpleNamespace(
            cuda_available=True,
            capability=(8, 9),
            device_index=0,
            device_name='fake',
            backend='nvidia_cuda',
            architecture='sm89',
        )
        mlp = MLPProviderResolution('off', 'off', 'synthetic')
        dense_resolution = object()
        attention = resolved(
            apply_module.ATTENTION_FP8_FLEX,
            dense_resolution=dense_resolution,
        )

        for request, expected_fallback in (
            (SPARSE_BACKEND_AUTO, True),
            (SPARSE_BACKEND_FLEX, False),
        ):
            with self.subTest(request=request):
                plan = H3OptimizationPlan(
                    sparse=SparseRequest(backend=request)
                )
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
                    return_value=(attention, self.qkv),
                ), mock.patch.object(
                    apply_module,
                    'configure_backend',
                    return_value=(object(), 50),
                ) as configure, mock.patch.object(
                    apply_module,
                    'install_dense_attention',
                    return_value=True,
                ) as install_dense, mock.patch.object(
                    apply_module,
                    '_install_mlp',
                    return_value=(mlp, 0),
                ), mock.patch.object(
                    apply_module,
                    '_ensure_sparse_runtime',
                    return_value=(object(), True),
                ), mock.patch.object(
                    apply_module,
                    'not_applicable_v_layout',
                    return_value=SimpleNamespace(
                        state='not_applicable',
                        reason='synthetic',
                        patched_blocks=0,
                    ),
                ):
                    apply_module.apply_plan(FakeModel(), plan)

                self.assertEqual(
                    configure.call_args.kwargs['backend_fallback_to_dense'],
                    expected_fallback,
                )
                if expected_fallback:
                    install_dense.assert_called_once_with(
                        mock.ANY,
                        dense_resolution,
                    )
                else:
                    install_dense.assert_not_called()

    def test_explicit_backend_status_is_not_called_a_fallback(self):
        plan = H3OptimizationPlan(
            sparse=SparseRequest(
                backend=SPARSE_BACKEND_TRITON,
                early_steps=2,
                early_kv=0.5,
                late_steps=2,
                late_kv=0.5,
            )
        )
        model = FakeModel(
            {
                PLAN_KEY: plan,
                'transformer_options': {
                    STATUS_KEY: {
                        'attention': {
                            'selected': apply_module.ATTENTION_TRITON_SPARSE,
                            'reason': 'explicit selection',
                        },
                        'sparse': {
                            'backend': SPARSE_BACKEND_TRITON,
                            'video_budget': 0.3,
                            'early_steps': 2,
                            'early_kv': 0.5,
                            'late_steps': 2,
                            'late_kv': 0.5,
                        },
                        'fused_qkv': {
                            'provider': 'standard_h3_qkv',
                            'reason': 'synthetic',
                        },
                        'mlp': {'provider': 'off'},
                    }
                },
            }
        )
        text = format_sparse_status(model)
        self.assertIn('Requested sparse backend: INT8 Triton', text)
        self.assertNotIn('Sparse fallback:', text)


if __name__ == '__main__':
    unittest.main()
