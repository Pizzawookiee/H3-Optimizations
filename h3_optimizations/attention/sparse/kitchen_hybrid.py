'''Budget-selected composition of the two native Sparse Kitchen geometries.'''

from __future__ import annotations

from dataclasses import dataclass

from .config import resolve_video_budget
from .kitchen_sparse import snapshot_for


QUALITY_GEOMETRY = '64q64kv'
FASTER_GEOMETRY = '64q128kv'
DEFAULT_SWITCH_BUDGET = 0.20


def _geometry_for(config, sequence, layer_index, transformer_options, threshold):
    snapshot = snapshot_for(transformer_options, sequence)
    budget = resolve_video_budget(
        config,
        snapshot.step_index,
        snapshot.total_steps,
        layer_index,
    )
    return FASTER_GEOMETRY if budget >= threshold else QUALITY_GEOMETRY


@dataclass
class HybridKitchenProjected:
    geometry: str
    projected: object

    def release(self):
        release = getattr(self.projected, 'release', None)
        if release is not None:
            release()
        self.projected = None


@dataclass
class HybridKitchenPrepared:
    geometry: str
    prepared: object

    def release(self):
        release = getattr(self.prepared, 'release', None)
        if release is not None:
            release()
        self.prepared = None


class HybridKitchenQKVProjector:
    name = 'hybrid_kitchen_qkv'

    def __init__(
        self,
        config,
        quality,
        faster,
        *,
        switch_budget=DEFAULT_SWITCH_BUDGET,
    ):
        self.config = config
        self.quality = quality
        self.faster = faster
        self.switch_budget = float(switch_budget)
        self.routing_summaries = bool(
            getattr(quality, 'routing_summaries', False)
            and getattr(faster, 'routing_summaries', False)
        )
        self.stream_output = bool(
            getattr(quality, 'stream_output', False)
            and getattr(faster, 'stream_output', False)
        )
        self.streamed_q = bool(
            getattr(quality, 'streamed_q', False)
            and getattr(faster, 'streamed_q', False)
        )
        self.strided_qk_input = bool(
            getattr(quality, 'strided_qk_input', False)
            and getattr(faster, 'strided_qk_input', False)
        )
        self.chunk_rows = int(quality.chunk_rows)
        if self.chunk_rows != int(faster.chunk_rows):
            raise ValueError('hybrid Kitchen projectors must use one chunk size')
        self.v_mode = quality.v_mode
        if self.v_mode != faster.v_mode:
            raise ValueError('hybrid Kitchen projectors must use one V mode')

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.switch_budget,
            self.quality.installation_signature,
            self.faster.installation_signature,
        )

    def bind(self, module):
        for projector in (self.quality, self.faster):
            bind = getattr(projector, 'bind', None)
            if bind is not None:
                bind(module)

    def try_project(
        self,
        module,
        x,
        rope_freqs,
        *,
        layer_index,
        transformer_options,
    ):
        geometry = _geometry_for(
            self.config,
            x.shape[0],
            layer_index,
            transformer_options,
            self.switch_budget,
        )
        projector = self.faster if geometry == FASTER_GEOMETRY else self.quality
        projected = projector.try_project(
            module,
            x,
            rope_freqs,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        if projected is None:
            return None
        return HybridKitchenProjected(geometry, projected)


class HybridSparseKitchenBackend:
    name = 'sparse_kitchen_int8_hybrid'
    approximate = True

    def __init__(
        self,
        config,
        quality,
        faster,
        projector,
        *,
        switch_budget=DEFAULT_SWITCH_BUDGET,
    ):
        self.config = config
        self.quality = quality
        self.faster = faster
        self.projector = projector
        self.switch_budget = float(switch_budget)
        self.stream_output = bool(quality.stream_output and faster.stream_output)
        self.release_carrier_before_out_proj = bool(
            quality.release_carrier_before_out_proj
            and faster.release_carrier_before_out_proj
        )
        if quality.output_layout != faster.output_layout:
            raise ValueError('hybrid Kitchen delegates must use one output layout')

    @property
    def output_layout(self):
        return self.quality.output_layout

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            self.switch_budget,
            self.quality.installation_signature,
            self.faster.installation_signature,
            (
                None
                if self.projector is None
                else self.projector.installation_signature
            ),
        )

    def _backend(self, geometry):
        if geometry == FASTER_GEOMETRY:
            return self.faster
        if geometry == QUALITY_GEOMETRY:
            return self.quality
        raise ValueError('unknown hybrid Kitchen geometry %r' % geometry)

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        geometry = _geometry_for(
            self.config,
            q.shape[-2],
            layer_index,
            transformer_options,
            self.switch_budget,
        )
        backend = self._backend(geometry)
        prepared = backend.prepare(
            q,
            k,
            v,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        return HybridKitchenPrepared(geometry, prepared)

    def prepare_projected(
        self,
        projected,
        *,
        layer_index,
        transformer_options,
    ):
        if not isinstance(projected, HybridKitchenProjected):
            raise TypeError('hybrid Kitchen received an invalid projected carrier')
        backend = self._backend(projected.geometry)
        prepared = backend.prepare_projected(
            projected.projected,
            layer_index=layer_index,
            transformer_options=transformer_options,
        )
        projected.projected = None
        return HybridKitchenPrepared(projected.geometry, prepared)

    def execute(self, prepared):
        if not isinstance(prepared, HybridKitchenPrepared):
            raise TypeError('hybrid Kitchen received an invalid prepared carrier')
        return self._backend(prepared.geometry).execute(prepared.prepared)

    def execute_projected(self, module, prepared):
        if not isinstance(prepared, HybridKitchenPrepared):
            raise TypeError('hybrid Kitchen received an invalid prepared carrier')
        return self._backend(prepared.geometry).execute_projected(
            module,
            prepared.prepared,
        )

    def execute_with_lse(self, prepared):
        if not isinstance(prepared, HybridKitchenPrepared):
            raise TypeError('hybrid Kitchen received an invalid prepared carrier')
        return self._backend(prepared.geometry).execute_with_lse(prepared.prepared)

    def as_status(self):
        status = self.quality.as_status()
        status.update({
            'backend': self.name,
            'sparse_architecture': 'comfy_kitchen_int8_hybrid',
            'sparse_q_tile': 64,
            'sparse_kv_tile': 64,
            'sparse_kv_tile_max': 128,
            'sparse_geometry_policy': (
                '64Q x 64KV below 20%; 64Q x 128KV at or above 20%'
            ),
            'hybrid_switch_budget': self.switch_budget,
            'qkv_projector': (
                None if self.projector is None else self.projector.name
            ),
        })
        return status
