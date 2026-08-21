'''Apply-plan composition through Memory then Sparse and the reverse.'''

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
    MemoryRequest,
    SparseRequest,
    STATUS_KEY,
    read_plan,
)
from h3_optimizations.qkv.providers import (  # noqa: E402
    MLPProviderResolution,
    QKVProviderResolution,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


class FakeModel:
    def __init__(self, options=None):
        self.model_options = deepcopy(options or {})
        self.object_patches = {}

    def clone(self):
        cloned = FakeModel(self.model_options)
        cloned.object_patches = dict(self.object_patches)
        return cloned


def resolved_attention(plan):
    selected = 'sparse_sage' if plan.sparse else 'comfy_kitchen_int8'
    return apply_module.ResolvedAttention(
        requested=selected,
        selected=selected,
        backend=SimpleNamespace(name=selected),
        reason='synthetic',
        backend_kind=selected,
        projector=None,
    )


def apply_in_order(base, first_request, second_request):
    plan = H3OptimizationPlan()
    if isinstance(first_request, MemoryRequest):
        plan = plan.with_memory(first_request)
    else:
        plan = plan.with_sparse(first_request)
    first = apply_module.apply_plan(base, plan)

    plan = read_plan(first)
    if isinstance(second_request, MemoryRequest):
        plan = plan.with_memory(second_request)
    else:
        plan = plan.with_sparse(second_request)
    return apply_module.apply_plan(first, plan)


class ApplyCompositionTests(unittest.TestCase):
    def test_both_node_orders_resolve_identically(self):
        memory = MemoryRequest()
        sparse = SparseRequest(
            video_budget=0.5,
            denser_early_late_steps=True,
        )
        inventory = SimpleNamespace(
            labels=lambda _name: (
                'TensorWiseINT8Layout+convrot256',
            ),
        )
        qkv = QKVProviderResolution(
            'standard_h3_qkv',
            False,
            'synthetic',
        )
        mlp = MLPProviderResolution(
            'generic_chunked_quantized',
            'mlp_chunked_native',
            'synthetic',
        )

        def resolve(plan, *_args):
            return resolved_attention(plan), qkv

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
            return_value=SimpleNamespace(
                cuda_available=True,
                capability=(12, 0),
                device_name='fake SM120',
                architecture='sm120',
            ),
        ), mock.patch.object(
            apply_module,
            '_resolve_dense',
            side_effect=resolve,
        ), mock.patch.object(
            apply_module,
            '_resolve_sparse',
            side_effect=resolve,
        ), mock.patch.object(
            apply_module,
            'configure_backend',
            return_value=(object(), 50),
        ), mock.patch.object(
            apply_module,
            'install_v_layout_compat',
            return_value=SimpleNamespace(
                state='installed',
                reason='synthetic',
                patched_blocks=50,
            ),
        ), mock.patch.object(
            apply_module,
            'install_dense_attention',
            return_value=True,
        ), mock.patch.object(
            apply_module,
            '_install_mlp',
            return_value=(mlp, 50),
        ), mock.patch.object(
            apply_module,
            '_ensure_sparse_runtime',
            return_value=(object(), True),
        ):
            left = apply_in_order(FakeModel(), memory, sparse)
            right = apply_in_order(FakeModel(), sparse, memory)

        self.assertEqual(
            read_plan(left).signature,
            read_plan(right).signature,
        )
        left_status = left.model_options[
            'transformer_options'
        ][STATUS_KEY]
        right_status = right.model_options[
            'transformer_options'
        ][STATUS_KEY]
        self.assertEqual(
            left_status['plan_signature'],
            right_status['plan_signature'],
        )
        self.assertTrue(
            left_status['sparse']['denser_early_late_steps']
        )


if __name__ == '__main__':
    unittest.main()
