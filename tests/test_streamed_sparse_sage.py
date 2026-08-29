'''CPU contracts for low-VRAM streamed Sparse Sage execution.'''

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
sys.path.insert(0, str(PACK))
sys.path.insert(0, str(ROOT))
TEST_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], '--cpu']

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_optimizations.attention.sparse.router import SparseTileRouter  # noqa: E402
import h3_optimizations.attention.sparse.sparse_sage_streamed as streamed  # noqa: E402
from h3_optimizations.normalized_rows import NormalizedRows  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def sparse_spec():
    return SimpleNamespace(
        signature=('test-sparse-spec',),
        q_tile=128,
        kv_tile=64,
        qk_format='block_int8',
        q_scale_layout='per_q_tile_float32',
        k_scale_layout='per_kv_tile_float32',
        projected_v_format='floating_hnd',
        summary_format='tile_mean',
        v_format='fp16',
        accumulator='f16',
        kernel=lambda *_args: None,
    )


def cpu_packer(x, output, scales, summaries, *, row_start, block_size):
    for block_start in range(0, x.shape[-2], block_size):
        block_end = min(block_start + block_size, x.shape[-2])
        value = x[..., block_start:block_end, :].float()
        scale = value.abs().amax(dim=(-2, -1)) / 127.0 + 1e-7
        quantized = value / scale[..., None, None]
        quantized += torch.where(quantized >= 0, 0.5, -0.5)
        output[
            ...,
            row_start + block_start:row_start + block_end,
            :,
        ].copy_(quantized.to(torch.int8))
        destination_block = (row_start + block_start) // block_size
        scales[..., destination_block].copy_(scale)
        summaries[..., destination_block, :].copy_(
            value.mean(dim=-2).to(summaries.dtype)
        )


class StreamedSparseSageTests(unittest.TestCase):
    def test_streamed_lut_chunks_reconstruct_existing_route_exactly(self):
        router = SparseTileRouter(q_tile=2, kv_tile=2)
        layout = SimpleNamespace(
            seq_len=12,
            video_range=(4, 12),
            segments=((0, 4, 'text'), (4, 12, 'video')),
            video_shape=(1, 2, 4),
            audio_t=0,
        )
        generator = torch.Generator().manual_seed(1234)
        q_summary = torch.randn((1, 2, 6, 4), generator=generator)
        k_summary = torch.randn((1, 2, 6, 4), generator=generator)

        expected_lut, expected_valid, expected_meta = (
            router.build_lut_from_summaries(
                q_summary,
                k_summary,
                layout,
                0.5,
            )
        )
        route_plan, actual_meta = streamed._prepare_streamed_route_plan(
            router,
            k_summary,
            layout,
            0.5,
        )
        pieces = []
        for tile_start in range(0, q_summary.shape[-2], 2):
            tile_end = min(tile_start + 2, q_summary.shape[-2])
            pieces.append(
                streamed._build_streamed_lut_chunk(
                    router,
                    route_plan,
                    q_summary[..., tile_start:tile_end, :],
                    tile_start=tile_start,
                )
            )
        actual_lut = torch.cat([piece[0] for piece in pieces], dim=2)
        actual_valid = torch.cat([piece[1] for piece in pieces], dim=2)

        self.assertTrue(torch.equal(actual_lut, expected_lut))
        self.assertTrue(torch.equal(actual_valid, expected_valid))
        self.assertEqual(actual_meta.as_dict(), expected_meta.as_dict())
        self.assertEqual(route_plan.k_summary.data_ptr(), k_summary.data_ptr())

    def test_projector_keeps_no_full_q_carrier(self):
        sequence = 300
        module = SimpleNamespace(heads=2, head_dim=128)
        x = torch.zeros((sequence, 256), dtype=torch.float32)
        rope = torch.arange(sequence).reshape(1, sequence, 1, 1, 1, 1)
        calls = []

        def project(_module, _x, rope_freqs, start, end):
            calls.append((start, end, tuple(rope_freqs[:, start:end].reshape(-1))))
            rows = torch.arange(start, end, dtype=torch.float32).view(1, 1, -1, 1)
            q = rows.expand(1, 2, -1, 128).clone()
            return q, q + 100, q + 200

        projected = streamed._assemble_streamed_sparse_qkv(
            module,
            x,
            rope,
            layer_index=3,
            spec=sparse_spec(),
            project_chunk_rows=128,
            query_chunk_rows=256,
            packer=cpu_packer,
            project_chunk=project,
        )

        self.assertEqual(
            [(start, end) for start, end, _rope in calls],
            [(0, 128), (128, 256), (256, 300)],
        )
        self.assertEqual(calls[-1][2], tuple(range(256, 300)))
        self.assertFalse(hasattr(projected, 'q_int8'))
        self.assertEqual(tuple(projected.k_int8.shape), (1, 2, sequence, 128))
        self.assertEqual(tuple(projected.k_scale.shape), (1, 2, 5))
        self.assertFalse(hasattr(projected, 'q_summary'))
        self.assertEqual(float(projected.k_summary[0, 0, -1, 0]), 377.5)
        self.assertEqual(float(projected.v[0, 0, -1, 0]), 499.0)
        self.assertEqual(projected.query_chunk_rows, 256)

    def test_projector_defaults_to_4096_query_rows(self):
        projector = streamed.StreamedSparseSageQKVProjector(sparse_spec())
        self.assertEqual(projector.chunk_rows, 4096)
        self.assertEqual(projector.query_chunk_rows, 4096)
        self.assertTrue(projector.streamed_q)

    def test_generic_source_projects_and_releases_one_bounded_q_slab(self):
        calls = []

        class FullQKVHeld:
            def __enter__(self):
                calls.append('enter')
                return self

            def __exit__(self, *_args):
                calls.append('exit')

            def project_hnd(self, x, _rope, start, end):
                calls.append(('project', start, end))
                rows = end - start
                q = (
                    x[start:end]
                    .reshape(rows, 2, 128)
                    .transpose(0, 1)
                    .unsqueeze(0)
                    .contiguous()
                )
                return q, q + 1, q + 2

        x = torch.arange(64 * 256, dtype=torch.float32).reshape(64, 256)
        projected = SimpleNamespace(
            module=SimpleNamespace(qkv_proj=object()),
            x=x,
            rope_freqs=None,
            projection_mode='native',
        )
        q_int8 = torch.empty((1, 2, 64, 128), dtype=torch.int8)
        q_scale = torch.empty((1, 2, 1), dtype=torch.float32)
        q_summary = torch.empty((1, 2, 1, 128), dtype=torch.float32)

        def factory(_module, _sample, mode):
            calls.append(('factory', mode))
            return FullQKVHeld()

        with mock.patch.object(
            streamed,
            'describe_linear',
            return_value=SimpleNamespace(convrot_int8_256=False),
        ):
            streamed._project_streamed_q_into(
                projected,
                0,
                64,
                q_int8,
                q_scale,
                q_summary,
                block_size=128,
                held_factory=factory,
                packer=cpu_packer,
            )

        self.assertEqual(
            calls,
            [('factory', 'native'), 'enter', ('project', 0, 64), 'exit'],
        )
        self.assertEqual(tuple(q_int8.shape), (1, 2, 64, 128))

    def test_execute_keeps_lazy_input_separate_from_attention_output(self):
        sequence = 128
        heads = 2
        hidden = heads * 128
        residual = torch.ones((sequence, hidden), dtype=torch.float32)
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, sequence, 0),),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        module = SimpleNamespace(
            heads=heads,
            head_dim=128,
            qkv_proj=object(),
            out_proj=lambda rows: rows,
        )
        projected = streamed.StreamedSparseSageQKV(
            module=module,
            x=source,
            rope_freqs=None,
            k_int8=torch.zeros(1, heads, sequence, 128, dtype=torch.int8),
            k_scale=torch.ones(1, heads, 2),
            v=None,
            k_summary=None,
            output_dtype=torch.float32,
            sequence=sequence,
            heads=heads,
            head_dim=128,
            layer_index=0,
            project_chunk_rows=128,
            query_chunk_rows=128,
            projection_mode='native',
        )
        route_plan = streamed.StreamedRoutePlan(
            geometry=SimpleNamespace(q_tiles=1),
            retained=1,
            k_summary=None,
            batch=1,
            heads=heads,
        )
        prepared = streamed.PreparedStreamedSparseSage(
            projected=projected,
            route_plan=route_plan,
            v_carrier=torch.empty(1),
            v_scale=torch.empty(1),
            metadata={},
        )

        class Held:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            @staticmethod
            def project_q_hnd(rows, _rope, start, stop):
                count = stop - start
                return (
                    rows[start:stop]
                    .reshape(count, heads, 128)
                    .transpose(0, 1)
                    .unsqueeze(0)
                    .contiguous()
                )

        def held_factory(_module, _sample, _mode):
            return Held()

        def dispatch(
            q_int8,
            _k_int8,
            _v,
            output,
            _lut,
            _valid,
            _threshold,
            q_scale,
            _k_scale,
            _v_scale,
            _dtype,
        ):
            output.copy_(
                q_int8.float() * q_scale[..., 0].unsqueeze(-1).unsqueeze(-1)
            )

        spec = SimpleNamespace(q_tile=128, dispatch=dispatch)
        backend = SimpleNamespace(
            executor=SimpleNamespace(spec=spec),
            router=object(),
        )
        project_q = streamed._project_streamed_q_into

        def project_with_cpu_fakes(*args, **kwargs):
            return project_q(
                *args,
                **kwargs,
                held_factory=held_factory,
                packer=cpu_packer,
            )

        source.output_buffer().fill_(-7)
        with mock.patch.object(
            streamed,
            '_project_streamed_q_into',
            side_effect=project_with_cpu_fakes,
        ), mock.patch.object(
            streamed,
            'describe_linear',
            return_value=SimpleNamespace(convrot_int8_256=False),
        ), mock.patch.object(
            streamed,
            '_build_streamed_lut_chunk',
            return_value=(torch.zeros(1), torch.ones(1)),
        ):
            actual = streamed.execute_streamed_sparse_sage(
                module,
                backend,
                prepared,
            )

        self.assertIs(actual, source.output_buffer())
        self.assertTrue(torch.equal(residual, source.materialize()))
        torch.testing.assert_close(actual, residual, rtol=2e-5, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
