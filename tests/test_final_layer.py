'''Production contracts for bounded H3 FinalLayer execution.'''

from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from h3_optimizations.memory import final_layer
import h3_optimizations.apply as apply_module
from h3_optimizations.plan import H3OptimizationPlan, MemoryRequest
from h3_optimizations.qkv.providers import MLPProviderResolution


class _Layer:
    norm = staticmethod(lambda value: value * 0.5)
    video_out = staticmethod(
        lambda value: value
        @ torch.arange(12, dtype=torch.float32).reshape(4, 3)
    )
    audio_out = staticmethod(
        lambda value: value
        @ torch.arange(8, dtype=torch.float32).reshape(4, 2)
    )

    @staticmethod
    def adaln_proj(_t_emb):
        shift = torch.tensor(
            [[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.float32
        )
        scale = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            dtype=torch.float32,
        )
        return shift, scale

    def forward(self, x, t_emb, video_seg, audio_seg):
        shift, scale = self.adaln_proj(t_emb)

        def project(segment, output):
            first, last, row = segment
            value = (
                self.norm(x[first:last]) * (1.0 + scale[row]) + shift[row]
            ).float()
            return output(value)

        return project(video_seg, self.video_out), project(
            audio_seg, self.audio_out
        )


class _Patcher:
    def __init__(self):
        self.object_patches = {}

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


class FinalLayerTests(unittest.TestCase):
    def _assert_matches_stock(self, segments):
        layer = _Layer()
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        expected = layer.forward(x, None, *segments)
        actual = final_layer.chunked_final_layer(
            layer, x, None, *segments, chunk_rows=3
        )
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

    def test_ragged_scalar_selectors_match_stock(self):
        self._assert_matches_stock(((0, 7, 0), (7, 11, 1)))

    def test_per_token_selectors_match_stock(self):
        self._assert_matches_stock((
            (0, 7, torch.tensor([0, 1, 0, 1, 1, 0, 1])),
            (7, 11, torch.tensor([1, 0, 0, 1])),
        ))

    def test_empty_stream_matches_stock(self):
        self._assert_matches_stock(((0, 11, 0), (11, 11, 1)))

    def test_install_is_owned_and_idempotent(self):
        patcher = _Patcher()
        model = SimpleNamespace(final_layer=_Layer())
        with mock.patch.object(
            final_layer, 'get_minimax_h3_model', return_value=model
        ):
            self.assertTrue(final_layer.install(patcher, 4096))
            self.assertFalse(final_layer.install(patcher, 4096))
            with self.assertRaises(final_layer.H3FinalLayerPatchError):
                final_layer.install(patcher, 2048)

    def test_memory_plan_installs_final_layer_even_when_mlp_is_off(self):
        patcher = _Patcher()
        plan = H3OptimizationPlan(
            memory=MemoryRequest(mlp_memory='off', chunk_rows=4096)
        )
        disabled = MLPProviderResolution('off', 'off', 'disabled')
        with mock.patch.object(
            apply_module, 'install_final_layer'
        ) as install, mock.patch.object(
            apply_module, 'resolve_mlp_provider', return_value=disabled
        ):
            resolution, patched_blocks = apply_module._install_mlp(
                patcher, plan, object(), object()
            )

        install.assert_called_once_with(patcher, 4096)
        self.assertIs(resolution, disabled)
        self.assertEqual(patched_blocks, 0)


if __name__ == '__main__':
    unittest.main()
