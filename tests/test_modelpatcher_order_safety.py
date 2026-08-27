'''Synthetic ModelPatcher permutation matrix for H3 composition hooks.'''

from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.patcher_extension  # noqa: E402
import h3_optimizations.apply as apply_module  # noqa: E402
from h3_optimizations.plan import (  # noqa: E402
    H3OptimizationPlan,
    MemoryRequest,
    PLAN_KEY,
    QKV_STREAMING_AUTO,
    QKV_STREAMING_FORCED,
    QKV_STREAMING_OFF,
    SparseRequest,
)

sys.argv = [sys.argv[0], *TEST_ARGS]


ATTENTION_KEY = 'diffusion_model.blocks.0.attn.forward'
BLOCK_KEY = 'diffusion_model.blocks.0.forward'
FINAL_KEY = 'diffusion_model.final_layer.forward'
H3_MARKER = '_synthetic_h3_owner'


def external_attention(value):
    return value * 11


def foreign_block(value):
    return value - 4


def foreign_attention(value):
    return value * 5


def foreign_final(value):
    return value + 7


def foreign_wrapper(executor, *args, **kwargs):
    return executor(*args, **kwargs)


class SyntheticModelPatcher:
    def __init__(self):
        self.model_options = {'transformer_options': {}}
        self.object_patches = {}
        self.patches = {}
        self.callbacks = {}
        self.wrappers = {}

    def clone(self):
        child = SyntheticModelPatcher()
        child.model_options = deepcopy(self.model_options)
        child.object_patches = self.object_patches.copy()
        child.patches = {key: value.copy() for key, value in self.patches.items()}
        child.callbacks = {
            call_type: {
                key: callbacks.copy()
                for key, callbacks in keyed.items()
            }
            for call_type, keyed in self.callbacks.items()
        }
        child.wrappers = {
            wrapper_type: {
                key: wrappers.copy()
                for key, wrappers in keyed.items()
            }
            for wrapper_type, keyed in self.wrappers.items()
        }
        for keyed in self.callbacks.get(
            comfy.patcher_extension.CallbacksMP.ON_CLONE,
            {},
        ).values():
            for callback in keyed:
                callback(self, child)
        return child

    def remove_callbacks_with_key(self, call_type, key):
        self.callbacks.get(call_type, {}).pop(key, None)

    def add_callback_with_key(self, call_type, key, callback):
        self.callbacks.setdefault(call_type, {})[key] = [callback]

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {})[key] = [wrapper]

    def execute(self):
        value = 1
        override = self.model_options['transformer_options'].get(
            'optimized_attention_override'
        )
        if override is not None:
            value = override(value)
        for key in (ATTENTION_KEY, BLOCK_KEY, FINAL_KEY):
            patch = self.object_patches.get(key)
            if patch is not None:
                value = patch(value)
        return value


def h3_callable(operation):
    def forward(value):
        return operation(value)

    setattr(forward, H3_MARKER, True)
    return forward


def synthetic_reconcile(patcher, plan, *, phase='node', force_rebuild=False):
    del force_rebuild
    patcher.model_options[PLAN_KEY] = plan
    options = patcher.model_options.setdefault('transformer_options', {})
    external = options.get('optimized_attention_override') is not None

    def install(key, operation):
        current = patcher.object_patches.get(key)
        if current is None or getattr(current, H3_MARKER, False):
            patcher.object_patches[key] = h3_callable(operation)

    if external:
        current = patcher.object_patches.get(ATTENTION_KEY)
        if getattr(current, H3_MARKER, False):
            patcher.object_patches.pop(ATTENTION_KEY)
    elif plan.memory is not None or plan.sparse is not None:
        install(ATTENTION_KEY, lambda value: value + 10)

    if plan.memory is not None:
        install(BLOCK_KEY, lambda value: value * 2)
        install(FINAL_KEY, lambda value: value + 3)

    options['h3_optimizations_status'] = {
        'plan_signature': plan.signature,
        'composition': {
            'phase': phase,
            'external_attention_preserved': external,
        },
    }
    apply_module._install_composition_hooks(patcher)
    return patcher


def downstream(model, kind):
    patched = model.clone()
    options = patched.model_options.setdefault('transformer_options', {})
    if kind in ('outer', 'diffusion', 'apply'):
        wrapper_type = {
            'outer': comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            'diffusion': comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            'apply': comfy.patcher_extension.WrappersMP.APPLY_MODEL,
        }[kind]
        options.setdefault('wrappers', {}).setdefault(wrapper_type, {})[
            'synthetic_foreign'
        ] = [foreign_wrapper]
    elif kind == 'attention_override':
        options['optimized_attention_override'] = external_attention
    elif kind == 'block_patch':
        patched.object_patches[BLOCK_KEY] = foreign_block
    elif kind == 'attention_patch':
        patched.object_patches[ATTENTION_KEY] = foreign_attention
    elif kind == 'final_patch':
        patched.object_patches[FINAL_KEY] = foreign_final
    elif kind == 'weight_patch':
        patched.patches['diffusion_model.weight'] = [('lora', 0.25)]
    elif kind == 'compile':
        patched.model_options['torch_compile_kwargs'] = {'backend': 'inductor'}
    return patched


def finalize(model):
    live = deepcopy(model.model_options)

    def executor(patcher, *_args, **_kwargs):
        return patcher.execute()

    wrapper = model.wrappers[
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING
    ][apply_module.PREPARE_WRAPPER_KEY][0]
    result = comfy.patcher_extension.WrapperExecutor.new_executor(
        executor,
        [wrapper],
    ).execute(
        model,
        None,
        None,
        model_options=live,
    )
    return result, live


def signature(model, live):
    wrappers = live['transformer_options'].get('wrappers', {})
    return {
        'object_patches': {
            key: getattr(value, H3_MARKER, False)
            for key, value in model.object_patches.items()
        },
        'foreign_wrapper_keys': {
            wrapper_type: tuple(sorted(keyed))
            for wrapper_type, keyed in wrappers.items()
            if 'synthetic_foreign' in keyed
        },
        'callback_count': len(model.callbacks[
            comfy.patcher_extension.CallbacksMP.ON_CLONE
        ][apply_module.CLONE_CALLBACK_KEY]),
        'prepare_count': len(model.wrappers[
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING
        ][apply_module.PREPARE_WRAPPER_KEY]),
        'external': live['transformer_options'].get(
            'optimized_attention_override'
        ) is external_attention,
        'weight_keys': tuple(sorted(model.patches)),
        'compiled': 'torch_compile_kwargs' in model.model_options,
    }


class ModelPatcherOrderSafetyTests(unittest.TestCase):
    def assert_order_equivalent(self, plan, modifier):
        before = apply_module.apply_plan(
            downstream(SyntheticModelPatcher(), modifier),
            plan,
        )
        after = downstream(
            apply_module.apply_plan(SyntheticModelPatcher(), plan),
            modifier,
        )
        before_result, before_live = finalize(before)
        after_result, after_live = finalize(after)
        self.assertEqual(before_result, after_result)
        self.assertEqual(
            signature(before, before_live),
            signature(after, after_live),
        )

    def test_kj_preview_outer_wrapper_double_is_order_equivalent(self):
        plan = H3OptimizationPlan(memory=MemoryRequest())
        with mock.patch.object(
            apply_module, 'is_minimax_h3', return_value=True
        ), mock.patch.object(
            apply_module, '_reconcile_plan', side_effect=synthetic_reconcile
        ):
            self.assert_order_equivalent(plan, 'outer')

    def test_unmarked_sla_override_double_keeps_full_q_semantics(self):
        plan = H3OptimizationPlan(
            memory=MemoryRequest(),
            sparse=SparseRequest(),
        )
        with mock.patch.object(
            apply_module, 'is_minimax_h3', return_value=True
        ), mock.patch.object(
            apply_module, '_reconcile_plan', side_effect=synthetic_reconcile
        ):
            self.assert_order_equivalent(plan, 'attention_override')

    def test_clone_reconstructs_h3_owned_execution_patch(self):
        plan = H3OptimizationPlan(memory=MemoryRequest())
        with mock.patch.object(
            apply_module, 'is_minimax_h3', return_value=True
        ), mock.patch.object(
            apply_module, '_reconcile_plan', side_effect=synthetic_reconcile
        ):
            parent = apply_module.apply_plan(SyntheticModelPatcher(), plan)
            parent_patch = parent.object_patches[ATTENTION_KEY]
            child = parent.clone()

        self.assertIsNot(child.object_patches[ATTENTION_KEY], parent_patch)
        self.assertTrue(
            getattr(child.object_patches[ATTENTION_KEY], H3_MARKER, False)
        )

    def test_downstream_modelpatcher_matrix_is_order_equivalent(self):
        plans = {
            'memory_off': H3OptimizationPlan(memory=MemoryRequest(
                qkv_streaming=QKV_STREAMING_OFF,
            )),
            'memory_auto': H3OptimizationPlan(memory=MemoryRequest(
                qkv_streaming=QKV_STREAMING_AUTO,
            )),
            'memory_forced': H3OptimizationPlan(memory=MemoryRequest(
                qkv_streaming=QKV_STREAMING_FORCED,
            )),
            'sparse': H3OptimizationPlan(sparse=SparseRequest()),
            'memory_sparse': H3OptimizationPlan(
                memory=MemoryRequest(),
                sparse=SparseRequest(),
            ),
            'advanced_sparse': H3OptimizationPlan(sparse=SparseRequest(
                early_steps=2,
                early_kv=0.5,
                late_steps=2,
                late_kv=0.5,
            )),
        }
        modifiers = (
            'clone',
            'outer',
            'diffusion',
            'apply',
            'attention_override',
            'block_patch',
            'attention_patch',
            'final_patch',
            'weight_patch',
            'compile',
        )

        with mock.patch.object(
            apply_module, 'is_minimax_h3', return_value=True
        ), mock.patch.object(
            apply_module, '_reconcile_plan', side_effect=synthetic_reconcile
        ):
            for plan_name, plan in plans.items():
                for modifier in modifiers:
                    with self.subTest(plan=plan_name, modifier=modifier):
                        self.assert_order_equivalent(plan, modifier)


if __name__ == '__main__':
    unittest.main()
