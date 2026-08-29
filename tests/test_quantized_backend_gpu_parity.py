"""Live SM89 numerical gates for the production attention backend matrix."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

TEST_ARGS = sys.argv[1:]
if not torch.cuda.is_available():
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()

from h3_optimizations.attention.sage_mem_eff import (  # noqa: E402
    SM89SageMemoryEfficientBackend,
)
from h3_optimizations.attention.sparse import triton_bf16  # noqa: E402
from h3_optimizations.attention.sparse.fp8_flex import (  # noqa: E402
    FLEX_BACKEND_TRITON,
    _per_head_scale,
    _quantize_fp8,
    block_mask_from_delta_lut,
    load_fp8_flex_spec,
)
from h3_optimizations.attention.sparse.frost_bf16 import (  # noqa: E402
    FrostBF16Executor,
)
from h3_optimizations.attention.sparse.sparse_quant import (  # noqa: E402
    quantize_qk as quantize_sparse_qk,
)
from h3_optimizations.attention.sparse.sparse_sage import (  # noqa: E402
    SparseSageError,
    SparseSageExecutor,
    load_sparse_sage_spec,
)
from h3_optimizations.dense_streamed_sage import (  # noqa: E402
    StreamedDenseSageBackend,
    StreamedDenseSageQKVProjector,
)
from h3_optimizations.normalized_rows import NormalizedRows  # noqa: E402
from h3_optimizations import native  # noqa: E402

sys.argv = [sys.argv[0], *TEST_ARGS]


def _relative_l2(actual, expected):
    delta = actual.float() - expected.float()
    return float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected.float()))


def _samples(sequence, heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            (1, sequence, heads, 128),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ).permute(0, 2, 1, 3)
        for _ in range(3)
    )


def _absolute_route(
    sequence,
    heads,
    q_tile,
    kv_tile,
    selected,
    *,
    compact=False,
    device="cuda",
):
    q_tiles = (sequence + q_tile - 1) // q_tile
    kv_tiles = (sequence + kv_tile - 1) // kv_tile
    selected = min(int(selected), kv_tiles)
    width = selected if compact else kv_tiles
    rows = []
    for head in range(heads):
        head_rows = []
        for q_block in range(q_tiles):
            blocks = sorted(
                (q_block + head + offset) % kv_tiles
                for offset in range(selected)
            )
            head_rows.append(blocks + [0] * (width - selected))
        rows.append(head_rows)
    route = torch.tensor([rows], dtype=torch.int32, device=device)
    counts = torch.full(
        (1, heads, q_tiles),
        selected,
        dtype=torch.int32,
        device=device,
    )
    return route, counts


def _delta_route(absolute):
    delta = absolute.clone()
    if delta.shape[-1] > 1:
        delta[..., 1:] -= absolute[..., :-1]
    return delta


def _routed_reference(q, k, v, route, counts, q_tile, kv_tile):
    sequence = int(q.shape[-2])
    mask = torch.zeros(
        (1, q.shape[1], sequence, sequence),
        dtype=torch.bool,
        device=q.device,
    )
    for head in range(q.shape[1]):
        for q_block in range(route.shape[-2]):
            q_start = q_block * q_tile
            q_stop = min(q_start + q_tile, sequence)
            for slot in range(int(counts[0, head, q_block])):
                kv_block = int(route[0, head, q_block, slot])
                kv_start = kv_block * kv_tile
                kv_stop = min(kv_start + kv_tile, sequence)
                mask[0, head, q_start:q_stop, kv_start:kv_stop] = True
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class _ProjectedQKV:
    def __init__(self, factory):
        self.factory = factory

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def project_kv_hnd(self, _x, _rope, start, stop):
        return (
            self.factory.k[..., start:stop, :].contiguous(),
            self.factory.v[..., start:stop, :].contiguous(),
        )

    @staticmethod
    def project_q_hnd(x, _rope, start, stop):
        rows = stop - start
        heads = int(x.shape[1]) // 128
        return (
            x[start:stop]
            .reshape(rows, heads, 128)
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )


class _ProjectedQKVFactory:
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __call__(self, _module, _sample, _projection_mode):
        return _ProjectedQKV(self)


SM89_ATTENTION_MATRIX = {
    "dense_sage": "test_dense_sage_q_is_independent_of_k_geometry_and_output_is_close",
    "streamed_dense_sage": "test_streamed_dense_sage_lazy_q_output_matches_dense_reference",
    "dense_kitchen": "test_kitchen_dense_and_sparse_outputs_are_close",
    "sparse_kitchen": "test_kitchen_dense_and_sparse_outputs_are_close",
    "sparse_sage": "test_sparse_sage_q_is_independent_of_k_geometry_and_output_is_close",
    "fp8_flex": "test_fp8_flex_chunking_is_exact_and_sparse_output_is_close",
    "frost_bf16": "test_frost_sparse_output_is_close",
    "triton_bf16": "test_bf16_triton_sparse_output_is_close",
}


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires an SM89 CUDA GPU",
)
class QuantizedBackendGPUParityTests(unittest.TestCase):
    def assertNumericallyClose(
        self,
        actual,
        expected,
        *,
        relative_l2,
        max_absolute,
    ):
        self.assertTrue(torch.isfinite(actual).all())
        delta = actual.float() - expected.float()
        relative_error = _relative_l2(actual, expected)
        absolute_error = float(delta.abs().max())
        self.assertLess(
            relative_error,
            relative_l2,
            "relative L2 %.8f" % relative_error,
        )
        self.assertLess(
            absolute_error,
            max_absolute,
            "max absolute error %.8f" % absolute_error,
        )

    def test_dense_sage_q_is_independent_of_k_geometry_and_output_is_close(self):
        q, k, v = _samples(128, 56, 20260853)
        backend = SM89SageMemoryEfficientBackend()
        paired_q, paired_scale, _k, _k_scale = backend.quantize_projected_qk(q, k)
        q_only, q_only_scale = backend.quantize_projected_q(q)
        self.assertTrue(torch.equal(q_only, paired_q))
        self.assertTrue(torch.equal(q_only_scale, paired_scale))

        actual = backend.execute(
            backend.prepare(q, k, v, layer_index=0, transformer_options={})
        )
        expected = F.scaled_dot_product_attention(q, k, v)
        self.assertNumericallyClose(
            actual,
            expected,
            relative_l2=0.05,
            max_absolute=0.25,
        )

    def test_streamed_dense_sage_lazy_q_output_matches_dense_reference(self):
        sequence = 256
        heads = 56
        q, k, v = _samples(sequence, heads, 20260857)
        source_rows = q.transpose(1, 2).reshape(sequence, heads * 128).contiguous()
        residual = source_rows.clone()
        source = NormalizedRows(
            residual,
            lambda rows: rows.clone(),
            ((0, sequence, 0),),
            None,
            None,
            lambda rows, _shift, _scale, _selector: rows,
        )
        out_scale = torch.linspace(
            0.75,
            1.25,
            heads * 128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        module = SimpleNamespace(
            heads=heads,
            head_dim=128,
            out_proj=lambda rows: rows * out_scale,
        )
        sage = SM89SageMemoryEfficientBackend()
        projector = StreamedDenseSageQKVProjector(
            sage,
            chunk_rows=128,
            projection_mode="native",
            held_factory=_ProjectedQKVFactory(k, v),
        )
        backend = StreamedDenseSageBackend(sage)
        projected = projector.project(
            module,
            source,
            None,
            layer_index=0,
            transformer_options={},
        )
        source.output_buffer().fill_(-7)
        prepared = backend.prepare_projected(
            projected,
            layer_index=0,
            transformer_options={},
        )
        actual = backend.execute_projected(module, prepared)
        expected = (
            F.scaled_dot_product_attention(q, k, v)
            .transpose(1, 2)
            .reshape(sequence, heads * 128)
        )
        expected.mul_(out_scale)

        self.assertIsNot(actual, residual)
        self.assertTrue(torch.equal(residual, source_rows))
        self.assertNumericallyClose(
            actual,
            expected,
            relative_l2=0.05,
            max_absolute=0.25,
        )

    def test_sparse_sage_q_is_independent_of_k_geometry_and_output_is_close(self):
        q, k, v = _samples(128, 56, 20260854)
        paired_q, paired_scale, _k, _k_scale = quantize_sparse_qk(q, k, 128, 64)
        q_only, q_only_scale, _dummy_k, _dummy_scale = quantize_sparse_qk(
            q,
            k[..., :1, :].contiguous(),
            128,
            64,
        )
        self.assertTrue(torch.equal(q_only, paired_q))
        self.assertTrue(torch.equal(q_only_scale, paired_scale))

        try:
            spec = load_sparse_sage_spec(capability=(8, 9))
        except SparseSageError as error:
            self.skipTest(str(error))
        route, valid = _absolute_route(
            128,
            56,
            spec.q_tile,
            spec.kv_tile,
            1,
        )
        lut = _delta_route(route)
        executor = SparseSageExecutor(spec)
        actual = executor.execute(
            executor.prepare(q, k, v, lut, valid, layer_index=0, metadata={})
        )
        expected = _routed_reference(
            q,
            k,
            v,
            route,
            valid,
            spec.q_tile,
            spec.kv_tile,
        )
        self.assertNumericallyClose(
            actual,
            expected,
            relative_l2=0.05,
            max_absolute=0.25,
        )

    def test_fp8_flex_chunking_is_exact_and_sparse_output_is_close(self):
        q, k, v = _samples(128, 2, 20260855)
        q_scale = _per_head_scale(q, 128)
        q_chunk_scale = _per_head_scale(q, 31)
        self.assertTrue(torch.equal(q_scale, q_chunk_scale))
        q_fp8 = _quantize_fp8(q, q_scale, 128)
        q_chunk_fp8 = _quantize_fp8(q, q_chunk_scale, 31)
        self.assertTrue(torch.equal(q_fp8, q_chunk_fp8))

        spec = load_fp8_flex_spec()
        k_scale = _per_head_scale(k, 31)
        v_scale = _per_head_scale(v, 31)
        k_fp8 = _quantize_fp8(k, k_scale, 31)
        v_fp8 = _quantize_fp8(v, v_scale, 31, column_major=True)
        route, valid = _absolute_route(
            128,
            2,
            spec.q_tile,
            spec.kv_tile,
            1,
        )
        lut = _delta_route(route)
        block_mask = block_mask_from_delta_lut(spec, lut, valid, 128)
        qk_scale = q_scale * k_scale

        def restore_qk_scale(score, batch, head, _q_index, _kv_index):
            return score.float() * qk_scale[batch, head]

        output = spec.attention(
            q_fp8,
            k_fp8,
            v_fp8,
            score_mod=restore_qk_scale,
            block_mask=block_mask,
            scale=128 ** -0.5,
            kernel_options={
                "BACKEND": FLEX_BACKEND_TRITON,
                "ROWS_GUARANTEED_SAFE": True,
                "BLOCK_M": spec.q_tile,
                "BLOCK_N": spec.kv_tile,
            },
        )
        actual = output.to(torch.bfloat16)
        actual.mul_(v_scale.to(torch.bfloat16)[..., None, None])
        expected = _routed_reference(
            q,
            k,
            v,
            route,
            valid,
            spec.q_tile,
            spec.kv_tile,
        )
        self.assertNumericallyClose(
            actual,
            expected,
            relative_l2=0.08,
            max_absolute=0.5,
        )

    def test_frost_sparse_output_is_close(self):
        q, k, v = _samples(128, 56, 20260856)
        executor = FrostBF16Executor()
        route, counts = _absolute_route(
            128,
            56,
            executor.spec.q_tile,
            executor.spec.kv_tile,
            1,
        )
        actual = executor.execute(
            executor.prepare(q, k, v, route, counts, layer_index=0, metadata={})
        )
        expected = _routed_reference(
            q,
            k,
            v,
            route,
            counts,
            executor.spec.q_tile,
            executor.spec.kv_tile,
        )
        self.assertNumericallyClose(
            actual,
            expected,
            relative_l2=0.02,
            max_absolute=0.02,
        )

    def test_kitchen_dense_and_sparse_outputs_are_close(self):
        if not native.is_available():
            self.skipTest(native.unavailable_reason())
        q, k, v = _samples(128, 2, 20260858)
        carrier = native.prequantize_int8_attention(q, k, v, cta_k=64)
        dense_actual = native.int8_attention_from_prequantized(carrier)
        dense_expected = F.scaled_dot_product_attention(q, k, v)
        self.assertNumericallyClose(
            dense_actual,
            dense_expected,
            relative_l2=0.05,
            max_absolute=0.25,
        )

        route, counts = _absolute_route(128, 2, 64, 64, 1)
        sparse_actual = native.block_sparse_int8_attention_from_prequantized(
            carrier,
            native.BlockSparseRoute(
                indices=route,
                counts=counts,
                q_tile=64,
                kv_tile=64,
                encoding="absolute",
            ),
        )
        sparse_expected = _routed_reference(q, k, v, route, counts, 64, 64)
        self.assertNumericallyClose(
            sparse_actual,
            sparse_expected,
            relative_l2=0.05,
            max_absolute=0.25,
        )

    def test_bf16_triton_sparse_output_is_close(self):
        if not triton_bf16.TRITON_AVAILABLE:
            self.skipTest("Triton is unavailable")
        q, k, v = _samples(128, 2, 20260859)
        route, counts = _absolute_route(
            128,
            2,
            triton_bf16.Q_TILE,
            triton_bf16.KV_TILE,
            1,
            compact=True,
        )
        actual = triton_bf16._launch(
            triton_bf16.PreparedTritonBF16(
                q=q,
                k=k,
                v=v,
                sparse_lut=route,
                dense_q_tiles=0,
                sparse_q_tiles=route.shape[-2],
                sparse_selected=route.shape[-1],
                layer_index=0,
                metadata={},
            )
        )
        expected = _routed_reference(
            q,
            k,
            v,
            route,
            counts,
            triton_bf16.Q_TILE,
            triton_bf16.KV_TILE,
        )
        self.assertNumericallyClose(
            actual,
            expected,
            relative_l2=0.01,
            max_absolute=0.02,
        )


class AttentionBackendMatrixContractTests(unittest.TestCase):
    def test_every_declared_sm89_backend_has_a_live_numerical_gate(self):
        methods = set(dir(QuantizedBackendGPUParityTests))
        self.assertEqual(
            set(SM89_ATTENTION_MATRIX),
            {
                "dense_sage",
                "streamed_dense_sage",
                "dense_kitchen",
                "sparse_kitchen",
                "sparse_sage",
                "fp8_flex",
                "frost_bf16",
                "triton_bf16",
            },
        )
        for backend, method in SM89_ATTENTION_MATRIX.items():
            with self.subTest(backend=backend):
                self.assertIn(method, methods)

    def test_delta_routes_round_trip_for_the_compared_sparse_rows(self):
        absolute, counts = _absolute_route(
            256,
            3,
            64,
            64,
            2,
            device="cpu",
        )
        decoded = _delta_route(absolute).cumsum(dim=-1)
        for head in range(absolute.shape[1]):
            for q_block in range(absolute.shape[2]):
                selected = int(counts[0, head, q_block])
                self.assertTrue(
                    torch.equal(
                        decoded[0, head, q_block, :selected],
                        absolute[0, head, q_block, :selected],
                    )
                )


if __name__ == "__main__":
    unittest.main()
