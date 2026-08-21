'''CPU tests for standalone sampler-step and packed-layout publication.'''

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))

from h3_optimizations.runtime.context import (  # noqa: E402
    H3RuntimeSession,
    RUNTIME_KEY,
    get_runtime_snapshot,
    make_diffusion_wrapper,
    make_outer_wrapper,
)


class RuntimeTests(unittest.TestCase):
    def test_sampler_callback_owns_step_progress(self):
        layout = SimpleNamespace(seq_len=384)
        options = {'sample_sigmas': torch.empty((11,))}
        session = H3RuntimeSession(strict_layout=True)
        token = session.begin_request(10)
        try:
            with patch(
                'h3_optimizations.runtime.context.resolve_layout',
                return_value=layout,
            ):
                first = session.observe(
                    [torch.zeros((1, 1))],
                    torch.zeros((1, 1)),
                    options,
                    {},
                )
                repeated = session.observe(
                    [torch.zeros((1, 1))],
                    torch.zeros((1, 1)),
                    options,
                    {},
                )
                session.complete_step(0, 10)
                second = session.observe(
                    [torch.zeros((1, 1))],
                    torch.zeros((1, 1)),
                    options,
                    {},
                )
        finally:
            session.end_request(token)
        self.assertEqual(
            [
                first.step_index,
                repeated.step_index,
                second.step_index,
            ],
            [0, 0, 1],
        )
        self.assertEqual(
            [
                first.total_steps,
                repeated.total_steps,
                second.total_steps,
            ],
            [10, 10, 10],
        )

    def test_outer_wrapper_preserves_and_extends_the_callback(self):
        session = H3RuntimeSession()
        observed = []

        def original_callback(step, _x0, _x, total_steps):
            observed.append((step, total_steps))

        def executor(
            _noise,
            _latent,
            _sampler,
            _sigmas,
            _mask,
            callback,
        ):
            self.assertEqual(session._step_index, 0)
            callback(0, None, None, 10)
            self.assertEqual(session._step_index, 1)
            return 'ok'

        result = make_outer_wrapper(session)(
            executor,
            None,
            None,
            None,
            torch.empty((11,)),
            None,
            original_callback,
        )
        self.assertEqual(result, 'ok')
        self.assertEqual(observed, [(0, 10)])
        self.assertEqual(session._step_index, -1)

    def test_session_publishes_only_package_owned_state(self):
        layout = SimpleNamespace(seq_len=384)
        options = {}
        session = H3RuntimeSession(strict_layout=True)
        with patch(
            'h3_optimizations.runtime.context.resolve_layout',
            return_value=layout,
        ):
            snapshot = session.observe(
                [torch.zeros((1, 1))],
                torch.zeros((1, 1)),
                options,
                {},
            )
        self.assertIs(snapshot.layout, layout)
        self.assertEqual(snapshot.step_index, -1)
        self.assertIs(options[RUNTIME_KEY], snapshot)
        self.assertIs(get_runtime_snapshot(options), snapshot)
        self.assertEqual(set(options), {'h3_optimizations_runtime'})

    def test_diffusion_wrapper_publishes_before_execution(self):
        layout = SimpleNamespace(seq_len=384)
        options = {}
        session = H3RuntimeSession(strict_layout=True)

        def executor(*_args, **_kwargs):
            self.assertIs(get_runtime_snapshot(options).layout, layout)
            return 'ok'

        wrapper = make_diffusion_wrapper(session)
        with patch(
            'h3_optimizations.runtime.context.resolve_layout',
            return_value=layout,
        ):
            result = wrapper(
                executor,
                [torch.zeros((1, 1))],
                torch.zeros((1,)),
                torch.zeros((1, 1)),
                options,
                minimax_payload={},
            )
        self.assertEqual(result, 'ok')


if __name__ == '__main__':
    unittest.main()
