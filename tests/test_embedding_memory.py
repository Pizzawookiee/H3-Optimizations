'''CPU equivalence and ownership tests for embedding-memory release.'''

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import torch  # noqa: E402
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import comfy.ops  # noqa: E402
from comfy.ldm.minimax import model as minimax  # noqa: E402
from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402
from h3_optimizations.memory import embedding  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


class Patcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}
        self.model_options = {'transformer_options': {}}

    def get_model_object(self, name):
        if name == 'diffusion_model':
            return self.model
        raise KeyError(name)

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


class EmbeddingMemoryTests(unittest.TestCase):
    def _model(self):
        model = MiniMaxH3Model(
            hidden_size=32,
            num_layers=0,
            token_refiner_num_layers=0,
            num_attention_heads=1,
            attention_head_dim=32,
            ffn_hidden_size=64,
            timestep_input_dim=16,
            time_embed_hidden_size=32,
            time_embed_dim=16,
            rope_inv_freq_len=4,
            dtype=torch.float32,
            device='cpu',
            operations=comfy.ops.disable_weight_init,
        )
        torch.manual_seed(41)
        for parameter in model.parameters():
            parameter.detach().copy_(torch.randn_like(parameter) * 0.02)
        model.rope.inv_freq.copy_(torch.rand_like(model.rope.inv_freq))
        return model

    def _inputs(self):
        video = torch.randn(1, 24, 2, 4, 4) * 0.1
        audio = torch.randn(1, 32, 2, 3) * 0.1
        reference = torch.randn(1, 24, 3, 4, 4) * 0.1
        context = torch.randn(1, 2, 32) * 0.1
        payload = {
            'visual_cond_noise_aug': 1.0,
            'refs': [{
                'kind': 'video',
                'latent_t': 3,
                'latent_h': 4,
                'latent_w': 4,
                'ref_audio_t': 0,
            }],
            'cond_video_latents': [reference],
        }
        return [video, audio], torch.tensor([500.0]), context, payload

    def test_release_forward_is_bit_identical_for_reference_video(self):
        model = self._model()
        x, timestep, context, payload = self._inputs()
        captured = []

        def capture(_module, args):
            captured.append(args[0].detach().clone())

        hook = model.final_layer.register_forward_pre_hook(capture)
        try:
            expected = model._forward(
                x,
                timestep,
                context,
                minimax_payload=payload,
            )
            actual = embedding.make_forward(model, model._forward)(
                x,
                timestep,
                context,
                minimax_payload=payload,
            )
        finally:
            hook.remove()

        self.assertEqual(len(captured), 2)
        self.assertTrue(torch.equal(captured[0], captured[1]))
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))

    def test_install_is_guarded_idempotent_and_clearable(self):
        patcher = Patcher(self._model())
        with mock.patch.object(embedding, '_validate_upstream_forward'):
            self.assertTrue(embedding.install(patcher))
            installed = patcher.object_patches[embedding.FORWARD_KEY]
            self.assertFalse(embedding.install(patcher))
            self.assertIs(patcher.object_patches[embedding.FORWARD_KEY], installed)
            self.assertTrue(embedding.clear(patcher))
        self.assertNotIn(embedding.FORWARD_KEY, patcher.object_patches)

    def test_install_recovers_original_from_applied_patch(self):
        model = self._model()
        first = Patcher(model)
        with mock.patch.object(embedding, '_validate_upstream_forward'):
            self.assertTrue(embedding.install(first))
            model._forward = first.object_patches[embedding.FORWARD_KEY]

            rebuilt = Patcher(model)
            self.assertTrue(embedding.install(rebuilt, force_rebuild=True))
        installed = rebuilt.object_patches[embedding.FORWARD_KEY]
        self.assertTrue(getattr(installed, embedding.OWNER_MARKER))
        self.assertIsNotNone(getattr(installed, embedding.ORIGINAL_MARKER))

    def test_real_upstream_auto_policy_matches_supported_comfy_generation(self):
        patcher = Patcher(self._model())
        installed = embedding.install(patcher)

        # Per-row mask support arrived with the Comfy H3 forward we currently
        # ship the static early-release implementation for. Older 0.33-era H3
        # implementations are intentionally recognized as stock-lifetime only.
        if hasattr(minimax, 'mask_row_values'):
            self.assertTrue(installed)
            self.assertTrue(embedding.is_installed(patcher))
            self.assertNotIn(
                embedding.FALLBACK_REASON_KEY,
                patcher.model_options['transformer_options'],
            )
        else:
            self.assertFalse(installed)
            self.assertFalse(embedding.is_installed(patcher))
            self.assertIn(
                'changed',
                patcher.model_options['transformer_options'][
                    embedding.FALLBACK_REASON_KEY
                ],
            )

    def test_foreign_forward_patch_is_preserved(self):
        patcher = Patcher(self._model())
        foreign = lambda *args, **kwargs: None
        patcher.object_patches[embedding.FORWARD_KEY] = foreign
        self.assertFalse(embedding.install(patcher))
        self.assertIs(patcher.object_patches[embedding.FORWARD_KEY], foreign)
        self.assertTrue(
            patcher.model_options['transformer_options'][
                'h3_optimizations_preserved_embedding_patch'
            ]
        )

    def test_incompatible_upstream_forward_uses_stock_lifetime(self):
        patcher = Patcher(self._model())
        with mock.patch.object(
            embedding.inspect,
            'getsource',
            return_value='def _forward(self, x):\n    return x\n',
        ):
            self.assertFalse(embedding.install(patcher))

        self.assertNotIn(embedding.FORWARD_KEY, patcher.object_patches)
        self.assertIn(
            'changed',
            patcher.model_options['transformer_options'][
                embedding.FALLBACK_REASON_KEY
            ],
        )

    def test_uninspectable_upstream_forward_uses_stock_lifetime(self):
        patcher = Patcher(self._model())
        with mock.patch.object(
            embedding.inspect,
            'getsource',
            side_effect=OSError('source unavailable'),
        ):
            self.assertFalse(embedding.install(patcher))

        self.assertNotIn(embedding.FORWARD_KEY, patcher.object_patches)
        self.assertIn(
            'cannot inspect',
            patcher.model_options['transformer_options'][
                embedding.FALLBACK_REASON_KEY
            ],
        )

    def test_explicit_release_rejects_incompatible_upstream_forward(self):
        patcher = Patcher(self._model())
        with mock.patch.object(
            embedding.inspect,
            'getsource',
            return_value='def _forward(self, x):\n    return x\n',
        ):
            with self.assertRaisesRegex(
                embedding.H3EmbeddingMemoryPatchError,
                'changed',
            ):
                embedding.install(patcher, strict=True)


if __name__ == '__main__':
    unittest.main()
