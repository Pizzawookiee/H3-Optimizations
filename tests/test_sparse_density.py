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
from h3_optimizations.nodes import H3SparseAttention  # noqa: E402
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

    def build_lut(self, _q, _k, _layout, budget):
        self.budgets.append(float(budget))
        return object(), object(), MaskMetadata(budget)

    def build_lut_from_summaries(
        self,
        _q_summary,
        _k_summary,
        _layout,
        budget,
    ):
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
            'enabled',
            'video_budget',
            'denser_early_late_steps',
        ],
        'density toggle is appended after existing serialized inputs',
    )
    denser = input_by_id(schema, 'denser_early_late_steps')
    check(
        denser.display_name == 'Denser Early/Late steps'
        and denser.default is False
        and '30 percentage points' in denser.tooltip,
        'density toggle is explicit and defaults off',
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
        and request.denser_early_late_steps is True,
        'node carries the enabled policy into the sparse request',
    )


def test_step_budgets():
    print('H3 Sparse Attention step budgets')
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
        'early/late video budget is capped at 100%',
    )
    check(
        resolve_video_budget(HybridSparseConfig(video_budget=0.5), 0, 20)
        == 0.5,
        'disabled policy preserves the configured budget',
    )


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
    test_step_budgets()
    test_runtime_step_resolution()
    print('\nall H3 sparse density tests passed')


if __name__ == '__main__':
    main()
