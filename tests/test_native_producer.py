"""The chunked INT8 attention producer, over the vendored library.

The property that matters is that a carrier built chunk by chunk is
**bit-identical** to one built from the whole tensor in a single call. That is
what makes the producer a drop-in: nothing downstream -- dense attention, the
sparse kernel, the router -- can tell which route assembled it, so nothing
downstream needs to care.

It is not obviously true. Chunking means the K anchor cannot be found by
scanning, so it is chosen up front from sampled rows and passed in, and chunk
boundaries have to land on tile boundaries or the per-thread scales cover a
different span. Both of those are places where "close enough" would pass a
tolerance test and still be wrong.

These need a GPU; see the note in test_kitchen_sparse.py about running the
suite with CUDA_VISIBLE_DEVICES=0.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_optimizations.native import int8_attention as native  # noqa: E402
from h3_optimizations.native import producer as P  # noqa: E402

requires_native = pytest.mark.skipif(
    not P.int8_attention_producer_is_available(),
    reason="needs the vendored INT8 attention library and a CUDA device",
)

BATCH, HEADS, SEQ, HEAD_DIM = 1, 4, 2048, 128


def _qkv(seed=3):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            BATCH, SEQ, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ).transpose(1, 2)
        for _ in range(3)
    )


def _produce(q, k, v, chunk_rows):
    spec = P.int8_attention_producer_spec(
        q.shape, k.shape, dtype=q.dtype, device=q.device
    )
    assert chunk_rows % spec.sequence_alignment == 0
    samples = k.index_select(
        2, torch.tensor(spec.k_anchor_positions, device=k.device)
    ).contiguous()
    anchor = P.select_int8_attention_k_anchor(spec, samples)
    producer = P.create_int8_attention_producer(spec, anchor)
    for start in range(0, SEQ, chunk_rows):
        end = min(start + chunk_rows, SEQ)
        P.quantize_int8_attention_qk_chunk(
            producer,
            q[:, :, start:end, :].contiguous(),
            k[:, :, start:end, :].contiguous(),
            q_start=start,
            k_start=start,
        )
    P.quantize_int8_attention_v(producer, v)
    return P.finalize_int8_attention_producer(producer)


@requires_native
@pytest.mark.parametrize("chunk_rows", [128, 512, 1024, 2048])
def test_chunked_carrier_is_identical_to_single_shot(chunk_rows):
    q, k, v = _qkv()
    chunked = _produce(q, k, v, chunk_rows)
    whole = native.prequantize_int8_attention(q, k, v, cta_k=chunked.cta_k)
    for name in ("q", "k", "v", "q_scale", "k_scale", "v_scale"):
        assert torch.equal(getattr(chunked, name), getattr(whole, name)), (
            "%s differs at chunk_rows=%d" % (name, chunk_rows)
        )


@requires_native
def test_a_produced_carrier_drives_dense_and_sparse_alike():
    q, k, v = _qkv()
    carrier = _produce(q, k, v, 1024)
    dense = native.int8_attention_from_prequantized(carrier)

    kv_tiles = (SEQ + carrier.cta_k - 1) // carrier.cta_k
    q_tiles = SEQ // native.Q_TILE
    indices = torch.arange(kv_tiles, dtype=torch.int32, device=q.device)
    route = native.BlockSparseRoute(
        indices=indices.view(1, 1, 1, -1)
        .expand(BATCH, HEADS, q_tiles, -1)
        .contiguous(),
        counts=torch.full(
            (BATCH, HEADS, q_tiles), kv_tiles, dtype=torch.int32, device=q.device
        ),
        q_tile=native.Q_TILE,
        kv_tile=carrier.cta_k,
    )
    routed = native.block_sparse_int8_attention_from_prequantized(carrier, route)
    assert torch.equal(routed, dense)


@requires_native
def test_misaligned_chunks_are_refused():
    """Silently accepting them would change the scales, not raise."""
    q, k, v = _qkv()
    spec = P.int8_attention_producer_spec(
        q.shape, k.shape, dtype=q.dtype, device=q.device
    )
    samples = k.index_select(
        2, torch.tensor(spec.k_anchor_positions, device=k.device)
    ).contiguous()
    producer = P.create_int8_attention_producer(
        spec, P.select_int8_attention_k_anchor(spec, samples)
    )
    offset = spec.sequence_alignment // 2
    with pytest.raises(ValueError, match="alignment"):
        P.quantize_int8_attention_qk_chunk(
            producer,
            q[:, :, offset : offset + spec.sequence_alignment, :].contiguous(),
            k[:, :, offset : offset + spec.sequence_alignment, :].contiguous(),
            q_start=offset,
            k_start=offset,
        )


@requires_native
def test_incomplete_coverage_is_refused():
    """A carrier with an unwritten hole must not reach a kernel."""
    q, k, v = _qkv()
    spec = P.int8_attention_producer_spec(
        q.shape, k.shape, dtype=q.dtype, device=q.device
    )
    samples = k.index_select(
        2, torch.tensor(spec.k_anchor_positions, device=k.device)
    ).contiguous()
    producer = P.create_int8_attention_producer(
        spec, P.select_int8_attention_k_anchor(spec, samples)
    )
    P.quantize_int8_attention_qk_chunk(
        producer,
        q[:, :, :1024, :].contiguous(),
        k[:, :, :1024, :].contiguous(),
        q_start=0,
        k_start=0,
    )
    P.quantize_int8_attention_v(producer, v)
    with pytest.raises(RuntimeError, match="do not cover"):
        P.finalize_int8_attention_producer(producer)


def test_the_producer_path_is_available_without_a_kitchen_release():
    """The failure this whole exercise exists to fix.

    kitchen_qkv was integrated against a Kitchen API that never shipped, so
    producer_api_available() was False everywhere and the fast QKV path
    silently never ran. It must not depend on what pip installed.
    """
    from h3_optimizations.kitchen_qkv import producer_api_available, resolve_kitchen

    if not P.int8_attention_producer_is_available():
        pytest.skip("needs the vendored INT8 attention library and a CUDA device")

    device = torch.device("cuda")
    assert producer_api_available(device=device)
    assert resolve_kitchen(device).__name__ == "h3_optimizations.native"
