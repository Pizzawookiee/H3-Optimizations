'''Opt-in contract for attention overrides that can consume streamed H3 Q.'''

from __future__ import annotations

import torch


STREAMED_H3_QKV_MARKER = 'supports_streamed_h3_qkv'
STREAMED_H3_QKV_CONSUMER = 'consume'


def get_streamed_h3_qkv_consumer(transformer_options):
    options = transformer_options or {}
    override = options.get('optimized_attention_override')
    if getattr(override, STREAMED_H3_QKV_MARKER, False) is not True:
        return None
    consumer = getattr(override, STREAMED_H3_QKV_CONSUMER, None)
    if not callable(consumer):
        raise TypeError(
            'optimized-attention override advertises %s=True but has no '
            'callable %s'
            % (STREAMED_H3_QKV_MARKER, STREAMED_H3_QKV_CONSUMER)
        )
    return consumer


def consume_streamed_h3_qkv(
    consumer,
    q_chunk,
    global_k,
    global_v,
    *,
    q_start,
    q_total,
    layer_index,
    transformer_options,
):
    out = consumer(
        q_chunk=q_chunk,
        global_k=global_k,
        global_v=global_v,
        q_start=q_start,
        q_total=q_total,
        layer_index=layer_index,
        transformer_options=transformer_options,
    )
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            'streamed H3 attention consumer returned %s; expected a tensor'
            % type(out).__name__
        )
    if out.ndim != 4 or out.shape != q_chunk.shape:
        raise RuntimeError(
            'streamed H3 attention consumer returned shape %s; expected HND %s'
            % (tuple(out.shape), tuple(q_chunk.shape))
        )
    if out.device != q_chunk.device:
        raise RuntimeError(
            'streamed H3 attention consumer returned %s output for %s input'
            % (out.device, q_chunk.device)
        )
    return out
