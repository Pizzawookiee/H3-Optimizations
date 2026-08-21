'''FP8 FlexAttention fallback for fixed-density H3 sparse routing.'''

from dataclasses import dataclass
import importlib
import inspect
import logging

import torch

import comfy.model_management

from .. import AttentionBackendUnavailable
from ...runtime.context import get_runtime_snapshot
from .config import HybridSparseConfig, resolve_video_budget
from .router import KV_TILE, Q_TILE, SparseRouterError, SparseTileRouter


CHUNK_ROWS = 4096
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FLEX_BACKEND_FLASH = 'FLASH'
FLEX_BACKEND_TRITON = 'TRITON'


class FP8FlexError(RuntimeError):
    pass


@dataclass(frozen=True)
class FP8FlexSpec:
    version: str
    attention: object
    block_mask_type: object
    kernel_backend: str = FLEX_BACKEND_TRITON
    fp8_dtype: torch.dtype = FP8_DTYPE
    q_tile: int = Q_TILE
    kv_tile: int = KV_TILE

    @property
    def signature(self):
        return (
            self.version,
            self.fp8_dtype,
            self.q_tile,
            self.kv_tile,
            self.kernel_backend,
            id(self.attention),
            id(self.block_mask_type),
        )


def load_fp8_flex_spec(kernel_backend=FLEX_BACKEND_TRITON):
    try:
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
    except ImportError as exc:
        raise FP8FlexError(
            'PyTorch FlexAttention is unavailable'
        ) from exc

    from_kv_blocks = getattr(BlockMask, 'from_kv_blocks', None)
    if not callable(from_kv_blocks):
        raise FP8FlexError(
            'PyTorch FlexAttention has no precomputed BlockMask API'
        )
    parameters = inspect.signature(from_kv_blocks).parameters
    required = {'BLOCK_SIZE', 'seq_lengths', 'compute_q_blocks'}
    if not required.issubset(parameters):
        raise FP8FlexError(
            'PyTorch FlexAttention BlockMask API is too old'
        )
    try:
        attention = torch.compile(flex_attention, fullgraph=True)
    except Exception as exc:
        raise FP8FlexError(
            'PyTorch FlexAttention compilation is unavailable'
        ) from exc
    return FP8FlexSpec(
        version=str(torch.__version__),
        attention=attention,
        block_mask_type=BlockMask,
        kernel_backend=str(kernel_backend),
    )


def _flash_attention_available():
    try:
        interface = importlib.import_module('flash_attn.cute.interface')
    except (ImportError, OSError):
        return False
    return callable(getattr(interface, '_flash_attn_fwd', None))


def select_flex_kernel_backend(capability, flash_available=None):
    if int(capability[0]) < 9:
        return FLEX_BACKEND_TRITON
    available = (
        _flash_attention_available()
        if flash_available is None
        else bool(flash_available())
    )
    return FLEX_BACKEND_FLASH if available else FLEX_BACKEND_TRITON


def preflight_fp8_flex(
    *,
    cuda_available,
    capability_getter,
    device=None,
    fp8_supported=None,
    dynamo_supported=None,
    flash_available=None,
    loader=None,
):
    if not cuda_available():
        raise FP8FlexError('FP8 FlexAttention requires NVIDIA CUDA')
    capability = capability_getter()
    if capability is None:
        raise FP8FlexError('FP8 FlexAttention GPU capability is unavailable')

    if fp8_supported is None:
        fp8_supported = lambda: comfy.model_management.supports_fp8_compute(
            device
        )
    try:
        supported = bool(fp8_supported())
    except Exception as exc:
        raise FP8FlexError(
            'FP8 FlexAttention capability probe failed: %s' % exc
        ) from exc
    if not supported:
        raise FP8FlexError(
            'FP8 compute is unsupported on device capability %d.%d'
            % (int(capability[0]), int(capability[1]))
        )

    if dynamo_supported is None:
        dynamo_supported = torch._dynamo.is_dynamo_supported
    if not dynamo_supported():
        raise FP8FlexError('PyTorch Dynamo is unavailable for FlexAttention')
    kernel_backend = select_flex_kernel_backend(
        capability,
        flash_available=flash_available,
    )
    if loader is not None:
        return loader(kernel_backend)
    return load_fp8_flex_spec(kernel_backend)


def block_mask_from_delta_lut(spec, lut, valid_block_num, sequence):
    sequence = int(sequence)
    expected = (
        lut.shape[0],
        lut.shape[1],
        (sequence + int(spec.q_tile) - 1) // int(spec.q_tile),
        (sequence + int(spec.kv_tile) - 1) // int(spec.kv_tile),
    )
    if tuple(lut.shape) != expected:
        raise FP8FlexError(
            'FlexAttention LUT shape %s does not match %s'
            % (tuple(lut.shape), expected)
        )
    if tuple(valid_block_num.shape) != expected[:-1]:
        raise FP8FlexError(
            'FlexAttention valid-count shape %s does not match %s'
            % (tuple(valid_block_num.shape), expected[:-1])
        )
    if lut.dtype != torch.int32 or valid_block_num.dtype != torch.int32:
        raise FP8FlexError('FlexAttention LUT and valid counts must be int32')
    if lut.device != valid_block_num.device:
        raise FP8FlexError('FlexAttention LUT and valid counts devices differ')

    kv_indices = torch.cumsum(lut, dim=-1, dtype=torch.int32)
    return spec.block_mask_type.from_kv_blocks(
        valid_block_num,
        kv_indices,
        BLOCK_SIZE=(int(spec.q_tile), int(spec.kv_tile)),
        seq_lengths=(sequence, sequence),
        compute_q_blocks=False,
    )


def _per_head_scale(x, chunk_rows):
    maximum = torch.zeros(
        x.shape[:2],
        dtype=torch.float32,
        device=x.device,
    )
    for start in range(0, x.shape[-2], chunk_rows):
        end = min(start + chunk_rows, x.shape[-2])
        chunk_maximum = x[..., start:end, :].abs().amax(
            dim=(-2, -1)
        ).to(torch.float32)
        maximum = torch.maximum(maximum, chunk_maximum)
    return (maximum / FP8_MAX).clamp_min(torch.finfo(x.dtype).tiny)


def _quantize_fp8(x, scale, chunk_rows, *, column_major=False):
    if column_major:
        storage = torch.empty(
            (*x.shape[:-2], x.shape[-1], x.shape[-2]),
            dtype=FP8_DTYPE,
            device=x.device,
        )
        output = storage.transpose(-2, -1)
    else:
        output = torch.empty(
            x.shape,
            dtype=FP8_DTYPE,
            device=x.device,
        )
    input_scale = scale.to(dtype=x.dtype)[..., None, None]
    for start in range(0, x.shape[-2], chunk_rows):
        end = min(start + chunk_rows, x.shape[-2])
        quantized = torch.clamp(
            x[..., start:end, :] / input_scale,
            min=-FP8_MAX,
            max=FP8_MAX,
        ).to(FP8_DTYPE)
        output[..., start:end, :].copy_(quantized)
    return output


@dataclass
class PreparedFP8Flex:
    q_fp8: torch.Tensor
    k_fp8: torch.Tensor
    v_fp8: torch.Tensor
    qk_scale: torch.Tensor
    v_scale: torch.Tensor
    block_mask: object
    output_dtype: torch.dtype
    output_shape: tuple
    layer_index: int
    sequence: int
    heads: int
    metadata: dict
    compile_signature: tuple


class FP8FlexBackend:
    name = 'flex_attention_fp8'
    requires_runtime_context = True
    approximate = True

    def __init__(
        self,
        config=None,
        *,
        spec=None,
        router=None,
        chunk_rows=CHUNK_ROWS,
        allow_cpu_for_tests=False,
    ):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError('config must be HybridSparseConfig')
        self.spec = spec if spec is not None else load_fp8_flex_spec()
        self.router = router or SparseTileRouter(
            self.config,
            q_tile=self.spec.q_tile,
            kv_tile=self.spec.kv_tile,
        )
        if (self.router.q_tile, self.router.kv_tile) != (
            self.spec.q_tile,
            self.spec.kv_tile,
        ):
            raise FP8FlexError(
                'router geometry %dQ x %dKV does not match Flex %dQ x %dKV'
                % (
                    self.router.q_tile,
                    self.router.kv_tile,
                    self.spec.q_tile,
                    self.spec.kv_tile,
                )
            )
        self.chunk_rows = int(chunk_rows)
        if self.chunk_rows <= 0:
            raise ValueError('FP8 Flex chunk rows must be positive')
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self._validated_signatures = set()
        self._unavailable_signatures = {}

    @property
    def installation_signature(self):
        return (
            self.name,
            self.config.signature,
            (type(self.router).__module__, type(self.router).__qualname__),
            self.spec.signature,
            self.chunk_rows,
        )

    @staticmethod
    def _snapshot(transformer_options, sequence):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            raise FP8FlexError(
                'FP8 FlexAttention requires an H3 runtime snapshot'
            )
        if not snapshot.valid_layout:
            raise FP8FlexError(
                'FP8 FlexAttention requires a valid packed layout: %s'
                % (snapshot.error or 'layout unavailable')
            )
        if int(snapshot.layout.seq_len) != int(sequence):
            raise FP8FlexError(
                'runtime layout sequence %d does not match attention sequence %d'
                % (snapshot.layout.seq_len, sequence)
            )
        return snapshot

    def _validate(self, q, k, v):
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise FP8FlexError(
                'FP8 FlexAttention requires equal HND rank-4 Q/K/V shapes'
            )
        batch, heads, sequence, head_dim = q.shape
        if batch != 1 or head_dim != 128:
            raise FP8FlexError(
                'FP8 FlexAttention requires batch 1 and head_dim 128'
            )
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or q.dtype != k.dtype
            or q.dtype != v.dtype
        ):
            raise FP8FlexError(
                'FP8 FlexAttention Q/K/V require matching fp16 or bf16 dtypes'
            )
        if q.device != k.device or q.device != v.device:
            raise FP8FlexError('FP8 FlexAttention Q/K/V devices differ')
        if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
            raise FP8FlexError(
                'FP8 FlexAttention Q/K/V last dimension must be contiguous'
            )
        if comfy.model_management.in_training:
            raise FP8FlexError('FP8 FlexAttention is inference-only')
        if not self.allow_cpu_for_tests and not q.is_cuda:
            raise FP8FlexError('FP8 FlexAttention requires CUDA')
        return heads, sequence

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        heads, sequence = self._validate(q, k, v)
        compile_signature = (
            str(q.device),
            q.dtype,
            tuple(q.shape),
            self.spec.kernel_backend,
        )
        unavailable = self._unavailable_signatures.get(compile_signature)
        if unavailable is not None:
            raise AttentionBackendUnavailable(unavailable)
        snapshot = self._snapshot(transformer_options, sequence)
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
        )
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                video_budget,
            )
        except SparseRouterError as exc:
            raise FP8FlexError('sparse routing failed: %s' % exc) from exc

        block_mask = block_mask_from_delta_lut(
            self.spec,
            lut,
            valid_block_num,
            sequence,
        )
        q_scale = _per_head_scale(q, self.chunk_rows)
        k_scale = _per_head_scale(k, self.chunk_rows)
        v_scale = _per_head_scale(v, self.chunk_rows)
        q_fp8 = _quantize_fp8(q, q_scale, self.chunk_rows)
        k_fp8 = _quantize_fp8(k, k_scale, self.chunk_rows)
        v_fp8 = _quantize_fp8(
            v,
            v_scale,
            self.chunk_rows,
            column_major=True,
        )
        metadata = mask_metadata.as_dict()
        metadata.update(
            {
                'layer': int(layer_index),
                'flex_attention_heads': int(heads),
                'qkv_projection': 'standard_qkv_fp8',
            }
        )
        return PreparedFP8Flex(
            q_fp8=q_fp8,
            k_fp8=k_fp8,
            v_fp8=v_fp8,
            qk_scale=q_scale * k_scale,
            v_scale=v_scale,
            block_mask=block_mask,
            output_dtype=q.dtype,
            output_shape=tuple(q.shape),
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            metadata=metadata,
            compile_signature=compile_signature,
        )

    def execute(self, prepared):
        qk_scale = prepared.qk_scale

        def restore_qk_scale(score, batch, head, _q_index, _kv_index):
            return score.float() * qk_scale[batch, head]

        kernel_options = {
            'BACKEND': self.spec.kernel_backend,
            'ROWS_GUARANTEED_SAFE': True,
        }
        if self.spec.kernel_backend == FLEX_BACKEND_TRITON:
            kernel_options.update(
                BLOCK_M=int(self.spec.q_tile),
                BLOCK_N=int(self.spec.kv_tile),
            )
        try:
            output = self.spec.attention(
                prepared.q_fp8,
                prepared.k_fp8,
                prepared.v_fp8,
                score_mod=restore_qk_scale,
                block_mask=prepared.block_mask,
                scale=prepared.q_fp8.shape[-1] ** -0.5,
                kernel_options=kernel_options,
            )
            if (
                tuple(output.shape) != prepared.output_shape
                or output.dtype != self.spec.fp8_dtype
                or output.device != prepared.q_fp8.device
            ):
                raise FP8FlexError(
                    'FlexAttention returned an invalid output contract'
                )
            output = output.to(prepared.output_dtype)
            output.mul_(
                prepared.v_scale.to(dtype=output.dtype)[..., None, None]
            )
        except Exception as exc:
            if prepared.compile_signature in self._validated_signatures:
                raise
            detail = str(exc).splitlines()[0]
            reason = (
                'FP8 FlexAttention %s failed before validation: %s'
                % (self.spec.kernel_backend, detail)
            )
            self._unavailable_signatures[prepared.compile_signature] = reason
            logging.warning('[H3 Optimizations] %s; using dense attention', reason)
            raise AttentionBackendUnavailable(reason) from exc
        self._validated_signatures.add(prepared.compile_signature)
        return output

    def requires_fallback_inputs(self, prepared):
        return prepared.compile_signature not in self._validated_signatures

    def as_status(self):
        return {
            'mode': self.name,
            'video_budget': float(self.config.video_budget),
            'denser_early_late_steps': bool(
                self.config.denser_early_late_steps
            ),
            'density_mode': self.config.density_mode,
            'flex_attention': self.spec.version,
            'sparse_q_tile': self.spec.q_tile,
            'sparse_kv_tile': self.spec.kv_tile,
            'kernel_backend': self.spec.kernel_backend,
            'qkv_dtype': str(self.spec.fp8_dtype),
            'qkv_scale_layout': 'per_head_float32',
            'output_dtype': 'fp16_or_bf16',
            'approximate': True,
        }
