"""Live SM89 numerical gates for every quantized sparse-attention carrier."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F


PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parents[1]
for _root in (str(PACK), str(ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from h3_optimizations.attention.sage_mem_eff import (  # noqa: E402
    SM89SageMemoryEfficientBackend,
)
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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9),
    "requires an SM89 CUDA GPU",
)
class QuantizedBackendGPUParityTests(unittest.TestCase):
    def assertNumericallyClose(self, actual, expected, *, relative_l2):
        self.assertTrue(torch.isfinite(actual).all())
        error = _relative_l2(actual, expected)
        self.assertLess(error, relative_l2, "relative L2 %.8f" % error)

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
        self.assertNumericallyClose(actual, expected, relative_l2=0.05)

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
        kv_tiles = 128 // spec.kv_tile
        lut = torch.arange(kv_tiles, dtype=torch.int32, device="cuda")
        lut[1:] = 1
        lut = lut.view(1, 1, 1, kv_tiles).expand(1, 56, 1, kv_tiles).contiguous()
        valid = torch.full((1, 56, 1), kv_tiles, dtype=torch.int32, device="cuda")
        executor = SparseSageExecutor(spec)
        actual = executor.execute(
            executor.prepare(q, k, v, lut, valid, layer_index=0, metadata={})
        )
        expected = F.scaled_dot_product_attention(q, k, v)
        self.assertNumericallyClose(actual, expected, relative_l2=0.05)

    def test_fp8_flex_chunking_is_exact_and_full_route_output_is_close(self):
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
        q_tiles = 128 // spec.q_tile
        kv_tiles = 128 // spec.kv_tile
        lut = torch.arange(kv_tiles, dtype=torch.int32, device="cuda")
        lut[1:] = 1
        lut = lut.view(1, 1, 1, kv_tiles).expand(1, 2, q_tiles, kv_tiles).contiguous()
        valid = torch.full((1, 2, q_tiles), kv_tiles, dtype=torch.int32, device="cuda")
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
        expected = F.scaled_dot_product_attention(q, k, v)
        self.assertNumericallyClose(actual, expected, relative_l2=0.08)

    def test_frost_full_route_output_is_close(self):
        q, k, v = _samples(128, 56, 20260856)
        route = torch.arange(2, dtype=torch.int32, device="cuda")
        route = route.view(1, 1, 1, 2).expand(1, 56, 2, 2).contiguous()
        counts = torch.full((1, 56, 2), 2, dtype=torch.int32, device="cuda")
        executor = FrostBF16Executor()
        actual = executor.execute(
            executor.prepare(q, k, v, route, counts, layer_index=0, metadata={})
        )
        expected = F.scaled_dot_product_attention(q, k, v)
        self.assertNumericallyClose(actual, expected, relative_l2=0.02)


if __name__ == "__main__":
    unittest.main()
