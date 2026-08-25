'''Benchmark-only query-sliced attention output and out-projection lifetime.'''

from dataclasses import dataclass, replace

import torch

from h3_optimizations import diagnostics


@dataclass(frozen=True)
class CapturedProjection:
    projected: object
    input: torch.Tensor


@dataclass
class PreparedStreamedOutput:
    prepared: object
    input: torch.Tensor


class CapturingProjector:
    '''Keep the disposable normalized block input beside the packed carrier.'''

    name = 'captured_chunked_kitchen_qkv'

    def __init__(self, projector):
        self.projector = projector

    @property
    def installation_signature(self):
        return (self.name, self.projector.installation_signature)

    def bind(self, module):
        bind = getattr(self.projector, 'bind', None)
        if bind is not None:
            bind(module)

    def try_project(self, module, x, rope_freqs, **kwargs):
        projected = self.projector.try_project(module, x, rope_freqs, **kwargs)
        return None if projected is None else CapturedProjection(projected, x)


class StreamedOutputBackend:
    '''Slice an existing carrier by query tile and reuse input for final output.

    This intentionally retains the full Q carrier.  It isolates the attention
    output/out_proj lifetime only; Q-only parity is a separate experiment.
    '''

    name = 'sparse_kitchen_streamed_output_experiment'

    def __init__(self, backend, query_chunk_rows):
        q_tile = int(backend.executor.q_tile)
        if q_tile not in (64, 128):
            raise ValueError('streamed output experiment requires 64- or 128-row Q tiles')
        if query_chunk_rows <= 0 or query_chunk_rows % 128:
            raise ValueError('query chunk rows must be a positive 128-row multiple')
        self.backend = backend
        self.query_chunk_rows = int(query_chunk_rows)
        self.requires_runtime_context = backend.requires_runtime_context
        self.approximate = backend.approximate

    @property
    def installation_signature(self):
        return (
            self.name, self.query_chunk_rows, self.backend.installation_signature
        )

    def as_status(self):
        status = self.backend.as_status()
        status.update({
            'streamed_attention_output': True,
            'query_chunk_rows': self.query_chunk_rows,
            'retains_full_q_carrier': True,
        })
        return status

    def prepare_projected(self, projected, **kwargs):
        if not isinstance(projected, CapturedProjection):
            raise TypeError('streamed output backend requires CapturedProjection')
        prepared = self.backend.prepare_projected(projected.projected, **kwargs)
        return PreparedStreamedOutput(prepared, projected.input)

    def execute_projected(self, module, wrapped):
        prepared = wrapped.prepared
        quantized = prepared.quantized
        route = prepared.route
        output = wrapped.input
        q_tile = int(route.q_tile)
        if int(quantized.q.shape[-1]) != 128:
            raise RuntimeError('streamed output experiment requires head_dim 128')
        sequence = int(quantized.q.shape[-2])
        packed_q_tiles = (sequence + 127) // 128
        if quantized.q_scale.shape[-1] % packed_q_tiles:
            raise RuntimeError('streamed output experiment received invalid Q scales')
        scales_per_packed_q_tile = quantized.q_scale.shape[-1] // packed_q_tiles

        for start in range(0, sequence, self.query_chunk_rows):
            stop = min(start + self.query_chunk_rows, sequence)
            first_tile = start // q_tile
            stop_tile = (stop + q_tile - 1) // q_tile
            first_scale_tile = start // 128
            stop_scale_tile = (stop + 127) // 128
            q = quantized.q[..., start:stop, :].contiguous()
            q_scale = quantized.q_scale[
                ...,
                first_scale_tile * scales_per_packed_q_tile:
                stop_scale_tile * scales_per_packed_q_tile,
            ].contiguous()
            chunk_carrier = replace(quantized, q=q, q_scale=q_scale)
            chunk_route = replace(
                route,
                indices=route.indices[..., first_tile:stop_tile, :].contiguous(),
                counts=route.counts[..., first_tile:stop_tile].contiguous(),
            )
            raw = self.backend.executor.kitchen.block_sparse_int8_attention_from_prequantized(
                chunk_carrier, chunk_route, output_layout='nhd'
            )
            flat = raw.transpose(1, 2).reshape(
                raw.shape[0], raw.shape[2], module.heads * module.head_dim
            )
            with diagnostics.stage('attention_out'):
                output[start:stop].copy_(module.out_proj(flat.squeeze(0)))
            del raw, flat, chunk_carrier, chunk_route, q, q_scale

        prepared.release()
        wrapped.prepared = None
        return output
