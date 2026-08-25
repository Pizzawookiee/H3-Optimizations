'''CPU contracts for the isolated H3 FinalLayer chunking experiment.'''

import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

import torch


BENCHMARK = Path(__file__).resolve().parents[1] / 'benchmarks' / 'bench_final_layer_chunking.py'
SPEC = importlib.util.spec_from_file_location('bench_final_layer_chunking', BENCHMARK)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class _Layer:
    norm = staticmethod(lambda value: value * 0.5)
    video_out = staticmethod(lambda value: value @ torch.arange(12, dtype=torch.float32).reshape(4, 3))
    audio_out = staticmethod(lambda value: value @ torch.arange(8, dtype=torch.float32).reshape(4, 2))

    @staticmethod
    def adaln_proj(_t_emb):
        shift = torch.tensor([[1, 2, 3, 4], [-1, -2, -3, -4]], dtype=torch.float32)
        scale = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
        return shift, scale


def stock(layer, x, t_emb, video_seg, audio_seg):
    shift, scale = layer.adaln_proj(t_emb)

    def project(segment, output):
        first, last, row = segment
        value = (layer.norm(x[first:last]) * (1 + scale[row]) + shift[row]).float()
        return output(value)

    return project(video_seg, layer.video_out), project(audio_seg, layer.audio_out)


class FinalLayerChunkingBenchmarkTests(unittest.TestCase):
    def test_chunking_is_exact_for_ragged_scalar_selectors(self):
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = ((0, 7, 0), (7, 11, 1))
        expected = stock(_Layer(), x, None, *segments)
        actual = bench.chunked_final_layer(_Layer(), x, None, *segments, 3)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

    def test_chunking_is_exact_for_per_token_selectors(self):
        x = torch.arange(44, dtype=torch.float32).reshape(11, 4)
        segments = (
            (0, 7, torch.tensor([0, 1, 0, 1, 1, 0, 1])),
            (7, 11, torch.tensor([1, 0, 0, 1])),
        )
        expected = stock(_Layer(), x, None, *segments)
        actual = bench.chunked_final_layer(_Layer(), x, None, *segments, 3)
        self.assertTrue(torch.allclose(expected[0], actual[0], atol=1e-4, rtol=0))
        self.assertTrue(torch.allclose(expected[1], actual[1], atol=1e-4, rtol=0))

    def test_gpu_acknowledgement_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                bench.parse_args([])


if __name__ == '__main__':
    unittest.main()
