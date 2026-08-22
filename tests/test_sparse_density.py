'''CPU-only contracts for the H3 Sparse Attention density policy.'''

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

sys.argv = [sys.argv[0], '--cpu']
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.backend import (  # noqa: E402
    HybridSparseBackend,
)
from h3_optimizations.attention.sparse.config import (  # noqa: E402
    HybridSparseConfig,
    resolve_video_budget,
)
from h3_optimizations.nodes import (  # noqa: E402
    H3SparseAttention,
    H3SparseAttentionAdvanced,
)
from h3_optimizations.runtime.context import (  # noqa: E402
    H3RuntimeSession,
    RUNTIME_KEY,
    RuntimeSnapshot,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print('  ok: %s' % message)


class TensorStub:
    shape = (1, 2, 384, 128)


class MaskMetadata:
    pure_video_q_tiles = 4

    def __init__(self, budget):
        self.budget = float(budget)

    def as_dict(self):
        return {'requested_video_budget': self.budget}


class RecordingRouter:
    def __init__(self):
        self.budgets = []

    def build_lut(self, _q, _k, _layout, budget, *, sink=None):
        del sink
        self.budgets.append(float(budget))
        return object(), object(), MaskMetadata(budget)

    def build_lut_from_summaries(
        self,
        _q_summary,
        _k_summary,
        _layout,
        budget,
        *,
        sink=None,
    ):
        del sink
        self.budgets.append(float(budget))
        return object(), object(), MaskMetadata(budget)


class RecordingExecutor:
    def prepare(
        self,
        _q,
        _k,
        _v,
        _lut,
        _valid_block_num,
        *,
        layer_index,
        metadata,
    ):
        return SimpleNamespace(layer_index=layer_index, metadata=metadata)

    def prepare_projected(
        self,
        _projected,
        _lut,
        _valid_block_num,
        *,
        metadata,
    ):
        return SimpleNamespace(metadata=metadata)


def make_backend(config):
    backend = object.__new__(HybridSparseBackend)
    backend.config = config
    backend.router = RecordingRouter()
    backend.executor = RecordingExecutor()
    backend.projector = None
    return backend


def options(step_index, total_steps=20):
    layout = SimpleNamespace(seq_len=384)
    snapshot = RuntimeSnapshot(
        request_id=0,
        step_index=step_index,
        total_steps=total_steps,
        layout=layout,
        compute_dtype=None,
        device=None,
    )
    return {RUNTIME_KEY: snapshot}


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


def test_node_schema_and_request():
    print('H3 Sparse Attention node policy')
    schema = H3SparseAttention.define_schema()
    check(
        [item.id for item in schema.inputs]
        == [
            'model',
            'video_budget',
            'denser_early_late_steps',
            'layer_video_budgets',
        ],
        'standard schema appends the optional static layer table',
    )
    denser = input_by_id(schema, 'denser_early_late_steps')
    check(
        denser.display_name == 'Denser Early/Late steps'
        and denser.default is False
        and '30 percentage points' in denser.tooltip,
        'legacy density toggle is explicit and defaults off',
    )

    model = SimpleNamespace(model_options={})
    patched = SimpleNamespace(model_options={})
    with mock.patch(
        'h3_optimizations.nodes.apply_plan',
        return_value=patched,
    ) as apply:
        result = H3SparseAttention.execute(
            model,
            video_budget=0.5,
            denser_early_late_steps=True,
        )
    request = apply.call_args.args[1].sparse
    check(
        result.args[0] is patched
        and request.video_budget == 0.5
        and request.backend == 'auto'
        and request.denser_early_late_steps is True
        and request.advanced_schedule is False,
        'standard node carries the legacy denser-step policy',
    )

    layer_budgets = ','.join(['0.1'] * 25 + ['0.3'] * 25)
    with mock.patch(
        'h3_optimizations.nodes.apply_plan',
        return_value=patched,
    ) as apply:
        H3SparseAttention.execute(
            model,
            video_budget=0.2,
            layer_video_budgets=layer_budgets,
        )
    request = apply.call_args.args[1].sparse
    check(
        request.layer_video_budgets[:25] == (0.1,) * 25
        and request.layer_video_budgets[25:] == (0.3,) * 25,
        'standard node parses exactly 50 static layer budgets',
    )


def test_advanced_node_schema_and_request():
    print('H3 Sparse Attention Advanced node policy')
    schema = H3SparseAttentionAdvanced.define_schema()
    check(
        [item.id for item in schema.inputs]
        == [
            'model',
            'video_budget',
            'early_steps',
            'early_kv',
            'late_steps',
            'late_kv',
            'backend',
        ],
        'advanced schema preserves existing controls and appends backend selection',
    )
    backend = input_by_id(schema, 'backend')
    check(
        backend.default == 'auto'
        and backend.options
        == ['auto', 'Sparse Sage', 'INT8 Triton', 'FP8 FlexAttention'],
        'advanced backend selector exposes the supported sparse backends',
    )
    check(
        input_by_id(schema, 'early_steps').default == 2
        and input_by_id(schema, 'early_kv').default == 0.5
        and input_by_id(schema, 'late_steps').default == 2
        and input_by_id(schema, 'late_kv').default == 0.5,
        'advanced early and late defaults match the public contract',
    )

    model = SimpleNamespace(model_options={})
    patched = SimpleNamespace(model_options={})
    with mock.patch(
        'h3_optimizations.nodes.apply_plan',
        return_value=patched,
    ) as apply:
        result = H3SparseAttentionAdvanced.execute(
            model,
            video_budget=0.3,
            early_steps=3,
            early_kv=0.6,
            late_steps=4,
            late_kv=0.7,
            backend='INT8 Triton',
        )
    request = apply.call_args.args[1].sparse
    check(
        result.args[0] is patched
        and request.backend == 'INT8 Triton'
        and request.video_budget == 0.3
        and request.early_steps == 3
        and request.early_kv == 0.6
        and request.late_steps == 4
        and request.late_kv == 0.7
        and request.denser_early_late_steps is False,
        'advanced node carries the backend and complete explicit schedule',
    )


def test_step_budgets():
    print('H3 Sparse Attention legacy step budgets')
    config = HybridSparseConfig(
        video_budget=0.5,
        denser_early_late_steps=True,
    )
    backend = make_backend(config)
    q = k = v = TensorStub()
    expected = {
        -1: 0.5,
        0: 0.8,
        1: 0.8,
        2: 0.5,
        17: 0.5,
        18: 0.8,
        19: 0.8,
    }
    for step_index, budget in expected.items():
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=0,
            transformer_options=options(step_index),
        )
        check(
            abs(
                prepared.sparse.metadata['requested_video_budget']
                - budget
            )
            < 1e-9,
            'step %d resolves to %.0f%% video budget'
            % (step_index, budget * 100.0),
        )

    projected = SimpleNamespace(
        sequence=384,
        q_summary=object(),
        k_summary=object(),
        heads=2,
    )
    prepared = backend.prepare_projected(
        projected,
        layer_index=0,
        transformer_options=options(18),
    )
    check(
        prepared.sparse.metadata['requested_video_budget'] == 0.8,
        'fused projected-QKV routing uses the same late-step policy',
    )
    check(
        resolve_video_budget(
            HybridSparseConfig(
                video_budget=0.85,
                denser_early_late_steps=True,
            ),
            0,
            20,
        )
        == 1.0,
        'legacy early/late video budget is capped at 100%',
    )
    check(
        resolve_video_budget(HybridSparseConfig(video_budget=0.5), 0, 20)
        == 0.5,
        'disabled legacy policy preserves the configured budget',
    )


def test_advanced_step_budgets():
    print('H3 Sparse Attention explicit step budgets')
    config = HybridSparseConfig(
        video_budget=0.3,
        early_steps=2,
        early_kv=0.5,
        late_steps=3,
        late_kv=0.7,
    )
    expected = {
        -1: 0.3,
        0: 0.5,
        1: 0.5,
        2: 0.3,
        16: 0.3,
        17: 0.7,
        18: 0.7,
        19: 0.7,
    }
    for step_index, budget in expected.items():
        check(
            resolve_video_budget(config, step_index, 20) == budget,
            'explicit step %d resolves to %.0f%% video budget'
            % (step_index, budget * 100.0),
        )

    check(
        resolve_video_budget(
            HybridSparseConfig(
                video_budget=0.7,
                early_steps=1,
                early_kv=0.2,
                late_steps=0,
                late_kv=0.4,
            ),
            0,
            20,
        )
        == 0.2,
        'explicit early KV may be lower than the middle-step budget',
    )
    check(
        resolve_video_budget(
            HybridSparseConfig(
                video_budget=0.3,
                early_steps=2,
                early_kv=0.4,
                late_steps=2,
                late_kv=0.6,
            ),
            1,
            3,
        )
        == 0.6,
        'overlapping early and late windows use the denser requested budget',
    )


def test_static_layer_budgets():
    print('H3 Sparse Attention static layer budgets')
    budgets = tuple(0.1 + layer_index * 0.01 for layer_index in range(50))
    config = HybridSparseConfig(
        video_budget=0.2,
        layer_video_budgets=budgets,
    )
    check(
        resolve_video_budget(config, 3, 20, 0) == 0.1
        and resolve_video_budget(config, 3, 20, 49) == 0.59,
        'static table resolves budgets by transformer layer',
    )

    backend = make_backend(config)
    q = k = v = TensorStub()
    backend.prepare(
        q,
        k,
        v,
        layer_index=7,
        transformer_options=options(3),
    )
    projected = SimpleNamespace(
        sequence=384,
        q_summary=object(),
        k_summary=object(),
        heads=2,
    )
    backend.prepare_projected(
        projected,
        layer_index=31,
        transformer_options=options(3),
    )
    check(
        backend.router.budgets == [0.17, 0.41000000000000003],
        'normal and projected routing both use the current layer budget',
    )
    try:
        resolve_video_budget(config, 3, 20)
    except ValueError as exc:
        check('layer_index is required' in str(exc), 'missing layer index fails clearly')
    else:
        raise AssertionError('missing layer index should fail')


def test_runtime_step_resolution():
    print('H3 runtime sampler-step publication')
    session = H3RuntimeSession()
    layout = SimpleNamespace(seq_len=384)
    context = SimpleNamespace(dtype=None)
    schedule = torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0])
    token = session.begin_request(4)
    with mock.patch(
        'h3_optimizations.runtime.context.resolve_layout',
        return_value=layout,
    ):
        try:
            first = session.observe(
                TensorStub(),
                context,
                {'sample_sigmas': schedule},
            )
            session.complete_step(1, 4)
            middle = session.observe(
                TensorStub(),
                context,
                {'sample_sigmas': schedule},
            )
        finally:
            session.end_request(token)
        unknown = session.observe(
            TensorStub(),
            context,
            {'sample_sigmas': schedule},
        )
    check(
        first.step_index == 0 and first.total_steps == 4,
        'request boundary publishes the first sampler step',
    )
    check(
        middle.step_index == 2 and middle.total_steps == 4,
        'sampler callback advances step publication',
    )
    check(
        unknown.step_index == -1 and unknown.total_steps == 4,
        'missing current-step metadata preserves the base-budget fallback',
    )


def main():
    test_node_schema_and_request()
    test_advanced_node_schema_and_request()
    test_step_budgets()
    test_advanced_step_budgets()
    test_static_layer_budgets()
    test_runtime_step_resolution()
    print('\nall H3 sparse density tests passed')


if __name__ == '__main__':
    main()
