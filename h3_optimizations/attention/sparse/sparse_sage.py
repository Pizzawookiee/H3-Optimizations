"""Portable Sparse Sage bindings for H3 hybrid attention."""

from dataclasses import dataclass, replace
import importlib
import importlib.metadata

import torch

from ..sage_v_fp8 import TRITON_AVAILABLE, direct_per_channel_fp8
from .router import KV_TILE, Q_TILE


class SparseSageError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseSageKernelSpec:
    """The complete ABI contract for one Sparse Sage architecture."""

    version: str
    architecture: str
    capability: tuple
    q_tile: int
    kv_tile: int
    v_format: str
    kernel: object
    accumulator: str
    v_quant_bound: float = 0.0
    extension_layout: str = "monolithic"
    fused_v_ops: object = None
    kernel_name: str = ""
    qk_format: str = "block_int8"
    q_scale_layout: str = "per_q_tile_float32"
    k_scale_layout: str = "per_kv_tile_float32"
    projected_v_format: str = "floating_hnd"
    summary_format: str = "tile_mean"

    @property
    def signature(self):
        return (
            str(self.version), str(self.architecture), tuple(self.capability), int(self.q_tile),
            int(self.kv_tile), str(self.v_format), str(self.accumulator),
            float(self.v_quant_bound), str(self.extension_layout),
            str(self.kernel_name), str(self.qk_format),
            str(self.q_scale_layout), str(self.k_scale_layout),
            str(self.projected_v_format), str(self.summary_format),
            id(self.kernel), id(self.fused_v_ops),
        )

    @property
    def uses_fp8_v(self):
        return self.v_format == "fp8"

    def validate_lut(self, lut, valid, *, batch, heads, sequence):
        expected = (
            int(batch), int(heads),
            (int(sequence) + self.q_tile - 1) // self.q_tile,
            (int(sequence) + self.kv_tile - 1) // self.kv_tile,
        )
        if tuple(lut.shape) != expected or tuple(valid.shape) != expected[:-1]:
            raise SparseSageError(
                "Sparse Sage LUT/valid shapes do not match %dx%d geometry"
                % (self.q_tile, self.kv_tile)
            )
        if lut.dtype != torch.int32 or valid.dtype != torch.int32:
            raise SparseSageError("Sparse Sage LUT and valid counts must be int32")
        if not lut.is_contiguous() or not valid.is_contiguous():
            raise SparseSageError("Sparse Sage LUT and valid counts must be contiguous")
        if lut.device != valid.device:
            raise SparseSageError("Sparse Sage LUT and valid counts devices differ")

    def quantize_qk(self, q, k):
        if q.is_cuda:
            try:
                from .sparse_quant import quantize_qk as quantize_qk_triton
            except Exception as exc:
                raise SparseSageError("Sparse Sage Q/K quantization requires Triton on CUDA") from exc
            return quantize_qk_triton(q, k, self.q_tile, self.kv_tile)
        return (
            *_quantize_blocks(q, self.q_tile),
            *_quantize_blocks(k, self.kv_tile, subtract_mean=True),
        )

    def prepare_v(self, v):
        if v.ndim != 4 or v.stride(-1) != 1:
            raise SparseSageError("V preparation requires an HND view with contiguous head dimension")
        if not self.uses_fp8_v:
            # The Ampere kernel's ABI is HND FP16 and has no V scale carrier.
            carrier = v.to(dtype=torch.float16).contiguous()
            if carrier.untyped_storage().data_ptr() == v.untyped_storage().data_ptr():
                carrier = carrier.clone(memory_format=torch.contiguous_format)
            return carrier, torch.empty((0,), dtype=torch.float32, device=v.device)
        if not v.is_cuda:
            raise SparseSageError("Sparse Sage FP8 V preparation requires CUDA")
        if TRITON_AVAILABLE:
            return direct_per_channel_fp8(
                v,
                scale_max=self.v_quant_bound,
                pad_to=128,
            )
        fused = self.fused_v_ops
        if fused is None:
            raise SparseSageError("Sparse Sage FP8 V preparation requires the _fused extension")
        b, h, sequence, dim = v.shape
        padded = (sequence + 127) // 128 * 128
        transposed = torch.empty((b, h, dim, padded), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, transposed, 1)
        try:
            v_fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
            v_scale = torch.empty((b, h, dim), dtype=torch.float32, device=v.device)
            fused.scale_fuse_quant_cuda(
                transposed, v_fp8, v_scale, sequence, self.v_quant_bound, 1
            )
        finally:
            del transposed
        return v_fp8, v_scale

    def dispatch(self, q_int8, k_int8, v, output, lut, valid, pv_threshold,
                 q_scale, k_scale, v_scale, output_dtype):
        # Sparge's Ampere ABI omits the FP8 V scale; Ada/Hopper fuse it.
        args = [q_int8, k_int8, v, output]
        if self.uses_fp8_v:
            args.extend((lut, valid, pv_threshold, q_scale, k_scale, v_scale))
        else:
            args.extend((lut, valid, pv_threshold, q_scale, k_scale))
        args.extend((1, 0, 1, 128 ** -0.5, 0))
        self.kernel(*args)


_AMPERE_KERNEL = "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold"
_SM90_KERNEL = "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90"
_SM89_F16_KERNEL = "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold"
_SM89_F32_KERNEL = "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold"

_SPLIT_QATTN = {
    (8, 0): ("sm80", "spas_sage_attn_qattn_sm80"),
    (8, 6): ("sm80", "spas_sage_attn_qattn_sm80"),
    (8, 7): ("sm80", "spas_sage_attn_qattn_sm80"),
    (8, 9): ("sm89", "spas_sage_attn_qattn_sm89"),
    (12, 0): ("sm89", "spas_sage_attn_qattn_sm89"),
    (9, 0): ("sm90", "spas_sage_attn_qattn_sm90"),
}

_SPLIT_KERNELS = {
    (8, 0): (_AMPERE_KERNEL,),
    (8, 6): (_AMPERE_KERNEL,),
    (8, 7): (_AMPERE_KERNEL,),
    (8, 9): (_SM89_F16_KERNEL, _SM89_F32_KERNEL),
    (9, 0): (_SM90_KERNEL,),
    (12, 0): (_SM89_F16_KERNEL,),
}


def _cuda_version():
    parts = (torch.version.cuda or "0.0").split(".")
    return int(parts[0]), int(parts[1])


def _kernel(surface, name):
    value = getattr(surface, name, None)
    return value if callable(value) else None


def _load_split_qattn_surface(capability):
    family, namespace = _SPLIT_QATTN[capability]
    module = importlib.import_module("spas_sage_attn._qattn_%s" % family)
    if any(_kernel(module, name) is not None for name in _SPLIT_KERNELS[capability]):
        return module, "split"
    return getattr(torch.ops, namespace), "split"


def _load_qattn_surface(capability):
    if capability == (12, 0):
        return _load_split_qattn_surface(capability)
    try:
        return importlib.import_module("spas_sage_attn._qattn"), "monolithic"
    except ModuleNotFoundError as exc:
        if exc.name != "spas_sage_attn._qattn":
            raise
    return _load_split_qattn_surface(capability)


def _load_fused_surface():
    module = importlib.import_module("spas_sage_attn._fused")
    if callable(getattr(module, "transpose_pad_permute_cuda", None)):
        surface = module
    else:
        surface = torch.ops.spas_sage_attn_fused
    if (not callable(getattr(surface, "transpose_pad_permute_cuda", None))
            or not callable(getattr(surface, "scale_fuse_quant_cuda", None))):
        raise SparseSageError("SpargeAttention's compiled _fused extension lacks V preparation ops")
    return surface


def resolve_sparse_sage_spec(qattn, fused, *, capability, version, cuda_version=None,
                             extension_layout="monolithic", sm90_v_quant_bound=2.25):
    """Resolve one exact architecture/symbol pair; never silently fallback."""
    capability = tuple(int(x) for x in capability)
    cuda_version = _cuda_version() if cuda_version is None else tuple(cuda_version)
    if capability in ((8, 0), (8, 6), (8, 7)):
        name = _AMPERE_KERNEL
        kernel = _kernel(qattn, name)
        if kernel is None:
            raise SparseSageError("Sparse Sage %s lacks required kernel %s" % ("SM%d%d" % capability, name))
        return SparseSageKernelSpec(
            version=str(version), architecture="sm%d%d" % capability,
            capability=capability, q_tile=128, kv_tile=64, v_format="fp16",
            kernel=kernel, accumulator="f16",
            extension_layout=extension_layout, kernel_name=name,
        )
    if capability == (8, 9):
        kernel = _kernel(qattn, _SM89_F16_KERNEL) if cuda_version >= (12, 8) else None
        name = _SM89_F16_KERNEL if kernel is not None else _SM89_F32_KERNEL
        if cuda_version >= (12, 8):
            accumulator = "f16" if kernel is not None else "f32"
        else:
            accumulator = "f32"
        if kernel is None:
            kernel = _kernel(qattn, name)
        if kernel is None:
            raise SparseSageError("Sparse Sage SM89 lacks required kernel %s" % name)
        if fused is None:
            raise SparseSageError("Sparse Sage SM89 requires the compiled _fused extension")
        return SparseSageKernelSpec(
            version=str(version), architecture="sm89", capability=capability,
            q_tile=128, kv_tile=64, v_format="fp8", kernel=kernel,
            accumulator=accumulator, v_quant_bound=2.25,
            extension_layout=extension_layout, fused_v_ops=fused,
            kernel_name=name,
        )
    if capability == (9, 0):
        name = _SM90_KERNEL
        kernel = _kernel(qattn, name)
        if kernel is None:
            raise SparseSageError("Sparse Sage SM90 lacks required kernel %s" % name)
        if fused is None:
            raise SparseSageError("Sparse Sage SM90 requires the compiled _fused extension")
        return SparseSageKernelSpec(
            version=str(version), architecture="sm90", capability=capability,
            q_tile=64, kv_tile=128, v_format="fp8", kernel=kernel,
            accumulator="f32", v_quant_bound=float(sm90_v_quant_bound),
            extension_layout=extension_layout, fused_v_ops=fused,
            kernel_name=name,
        )
    if capability == (12, 0):
        if cuda_version < (12, 8):
            raise SparseSageError(
                "Sparse Sage SM120 requires CUDA 12.8 or newer; found CUDA %d.%d"
                % cuda_version
            )
        name = _SM89_F16_KERNEL
        kernel = _kernel(qattn, name)
        if kernel is None:
            raise SparseSageError("Sparse Sage SM120 lacks required kernel %s" % name)
        if fused is None:
            raise SparseSageError("Sparse Sage SM120 requires the compiled _fused extension")
        return SparseSageKernelSpec(
            version=str(version), architecture="sm120", capability=capability,
            q_tile=128, kv_tile=64, v_format="fp8", kernel=kernel,
            accumulator="f16", v_quant_bound=2.25,
            extension_layout=extension_layout, fused_v_ops=fused,
            kernel_name=name,
        )
    raise SparseSageError("Sparse Sage does not support device capability %d.%d" % capability)


def load_sparse_sage_spec(*, capability=None, capability_getter=None, cuda_version=None):
    """Load current SpargeAttention extension surfaces and resolve its ABI."""
    try:
        importlib.import_module("spas_sage_attn")
    except Exception as exc:
        raise SparseSageError(
            "Hybrid Sparse Attention requires the compiled spas_sage_attn package"
        ) from exc
    if capability is None:
        getter = capability_getter or torch.cuda.get_device_capability
        capability = tuple(getter())
    if capability is not None:
        capability = tuple(int(x) for x in capability)
        if capability not in ((8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (12, 0)):
            raise SparseSageError(
                "Sparse Sage does not support device capability %d.%d" % capability
            )
    try:
        qattn, extension_layout = _load_qattn_surface(capability)
    except Exception as exc:
        raise SparseSageError(
            "Hybrid Sparse Attention requires SpargeAttention's compiled "
            "attention extension for SM%d%d" % capability
        ) from exc
    try:
        version = importlib.metadata.version("spas-sage-attn")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    try:
        fused = _load_fused_surface()
    except Exception:
        fused = None
    spec = resolve_sparse_sage_spec(
        qattn, fused, capability=capability, version=version,
        cuda_version=cuda_version, extension_layout=extension_layout,
        sm90_v_quant_bound=448.0 if extension_layout == "split" else 2.25,
    )
    return spec


def preflight_sparse_sage(spec_loader=load_sparse_sage_spec, cuda_available=None,
                          capability_getter=None):
    if cuda_available is None:
        cuda_available = torch.cuda.is_available
    if not cuda_available():
        raise SparseSageError("Hybrid Sparse Attention requires CUDA")
    capability_getter = capability_getter or torch.cuda.get_device_capability
    capability = tuple(capability_getter())
    return spec_loader(capability=capability, capability_getter=capability_getter)


def _quantize_blocks(x, block, *, subtract_mean=False):
    if x.ndim != 4 or x.stride(-1) != 1:
        raise SparseSageError("Q/K quantization requires HND views with contiguous head dimension")
    b, h, sequence, dim = x.shape
    blocks = (sequence + block - 1) // block
    output = torch.empty((b, h, sequence, dim), dtype=torch.int8, device=x.device)
    scales = torch.empty((b, h, blocks), dtype=torch.float32, device=x.device)
    mean = x.mean(dim=-2, keepdim=True).float() if subtract_mean else None
    for index in range(blocks):
        start, stop = index * block, min(index * block + block, sequence)
        value = x[..., start:stop, :].float()
        if mean is not None:
            value = value - mean
        scale = value.abs().amax(dim=(-2, -1)) / 127.0 + 1e-7
        quantized = value / scale[..., None, None]
        quantized = quantized + torch.where(quantized >= 0, 0.5, -0.5)
        output[..., start:stop, :].copy_(quantized.to(torch.int8))
        scales[..., index].copy_(scale)
    return output, scales


def quantize_qk(q, k, q_tile=Q_TILE, kv_tile=KV_TILE):
    if q.is_cuda:
        try:
            from .sparse_quant import quantize_qk as quantize_qk_triton
        except Exception as exc:
            raise SparseSageError("Sparse Sage Q/K quantization requires Triton on CUDA") from exc
        return quantize_qk_triton(q, k, q_tile, kv_tile)
    q_int8, q_scale = _quantize_blocks(q, q_tile)
    k_int8, k_scale = _quantize_blocks(k, kv_tile, subtract_mean=True)
    return q_int8, q_scale, k_int8, k_scale


def prepare_v_fp8(v, fused=None):
    spec = SparseSageKernelSpec(
        version="unknown", architecture="sm89", capability=(8, 9),
        q_tile=Q_TILE, kv_tile=KV_TILE, v_format="fp8", kernel=None,
        accumulator="f32", v_quant_bound=2.25, fused_v_ops=fused,
    )
    return spec.prepare_v(v)


@torch.library.custom_op("h3_optimizations::prepare_sparse_sage_v", mutates_args=(), device_types="cuda")
def prepare_sparse_sage_v_op(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spec = load_sparse_sage_spec(
        capability=torch.cuda.get_device_capability(v.device)
    )
    return spec.prepare_v(v)


@prepare_sparse_sage_v_op.register_fake
def _prepare_sparse_sage_v_fake(v):
    padded = (v.shape[-2] + 127) // 128 * 128
    return v.new_empty((*v.shape[:-2], v.shape[-1], padded), dtype=torch.float8_e4m3fn), v.new_empty((*v.shape[:-2], v.shape[-1]), dtype=torch.float32)


def prepare_sparse_sage_v(v, _fused=None):
    return prepare_sparse_sage_v_op(v)


@torch.library.custom_op("h3_optimizations::sparse_sage_attention", mutates_args=(), device_types="cuda")
def sparse_sage_attention_op(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_carrier: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    pv_threshold: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    spec = load_sparse_sage_spec(
        capability=torch.cuda.get_device_capability(q_int8.device)
    )
    output = torch.empty(q_int8.shape, dtype=output_dtype, device=q_int8.device)
    spec.dispatch(q_int8, k_int8, v_carrier, output, lut, valid_block_num,
                  pv_threshold, q_scale, k_scale, v_scale, output_dtype)
    return output


@sparse_sage_attention_op.register_fake
def _sparse_sage_attention_fake(q_int8, k_int8, v_carrier, lut, valid_block_num,
                                pv_threshold, q_scale, k_scale, v_scale,
                                output_dtype):
    return q_int8.new_empty(q_int8.shape, dtype=output_dtype)


@dataclass
class PreparedSparseSage:
    q_int8: torch.Tensor
    k_int8: torch.Tensor
    v_carrier: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    lut: torch.Tensor
    valid_block_num: torch.Tensor
    output_dtype: torch.dtype
    output_shape: tuple
    layer_index: int
    sequence: int
    heads: int
    metadata: dict
    pv_threshold: torch.Tensor


class SparseSageExecutor:
    def __init__(self, spec, *, allow_cpu_for_tests=False, qk_quantizer=None,
                 v_preparer=None, low_level_selector=None):
        if not isinstance(spec, SparseSageKernelSpec):
            raise TypeError("SparseSageExecutor requires a preflighted SparseSageKernelSpec")
        self.spec = spec
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self.qk_quantizer = qk_quantizer or self.spec.quantize_qk
        self.v_preparer = v_preparer
        self.low_level_selector = low_level_selector or self._select_low_level
        self._use_sparse_sage_op = low_level_selector is None and not self.allow_cpu_for_tests

    def _select_low_level(self, _q):
        return self.spec.kernel

    def _prepare_v(self, v):
        if self.v_preparer is not None:
            return self.v_preparer(v, self.spec.fused_v_ops)
        if self.allow_cpu_for_tests:
            return self.spec.prepare_v(v)
        return prepare_sparse_sage_v(v)

    def _validate(self, q, k, v, lut, valid):
        if q.shape != k.shape or q.shape != v.shape:
            raise SparseSageError("Sparse Sage requires equal self-attention Q/K/V shapes; got %s %s %s" % (tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        if q.ndim != 4:
            raise SparseSageError("Sparse Sage expects HND rank-4 tensors")
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise SparseSageError("released H3 expects attention batch 1; got %d" % batch)
        if head_dim != 128:
            raise SparseSageError("Sparse Sage requires head_dim 128; got %d" % head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16) or q.dtype != k.dtype or q.dtype != v.dtype:
            raise SparseSageError("Sparse Sage Q/K/V require matching fp16 or bf16 dtypes")
        if q.device != k.device or q.device != v.device:
            raise SparseSageError("Sparse Sage Q/K/V devices differ")
        if any(t.stride(-1) != 1 for t in (q, k, v)):
            raise SparseSageError("Sparse Sage Q/K/V last dimension must be contiguous")
        self.spec.validate_lut(lut, valid, batch=batch, heads=heads, sequence=sequence)
        if lut.device != q.device:
            raise SparseSageError("Sparse Sage LUT and Q/K/V devices differ")
        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise SparseSageError("Sparse Sage requires CUDA")
            capability = tuple(torch.cuda.get_device_capability(q.device))
            if capability != self.spec.capability:
                raise SparseSageError(
                    "Sparse Sage resolved %s but execution device is %d.%d"
                    % (self.spec.architecture, capability[0], capability[1])
                )
        return heads, sequence

    def _validate_v(self, v, carrier, scale, sequence):
        if self.spec.uses_fp8_v:
            expected = (v.shape[0], v.shape[1], v.shape[3], (sequence + 127) // 128 * 128)
            if (tuple(carrier.shape) != expected or carrier.dtype != torch.float8_e4m3fn
                    or carrier.device != v.device or not carrier.is_contiguous()
                    or scale is None):
                raise SparseSageError("V preparer returned an invalid FP8 carrier")
            if (tuple(scale.shape) != (v.shape[0], v.shape[1], v.shape[3])
                    or scale.dtype != torch.float32 or scale.device != v.device
                    or not scale.is_contiguous()):
                raise SparseSageError("V preparer returned an invalid FP8 scale")
        else:
            if (tuple(carrier.shape) != tuple(v.shape) or carrier.dtype != torch.float16
                    or carrier.device != v.device or not carrier.is_contiguous()):
                raise SparseSageError("Ampere Sparse Sage requires contiguous FP16 HND V")
            if (not torch.is_tensor(scale) or scale.numel() != 0
                    or scale.dtype != torch.float32 or scale.device != v.device):
                raise SparseSageError("Ampere Sparse Sage V must not carry an FP8 scale")

    def _metadata(self, metadata):
        details = dict(metadata)
        details.update({
            "sparse_architecture": self.spec.architecture,
            "sparse_q_tile": self.spec.q_tile,
            "sparse_kv_tile": self.spec.kv_tile,
            "sparse_v_format": self.spec.v_format,
            "sparse_accumulator": self.spec.accumulator,
            "sparse_v_quant_bound": self.spec.v_quant_bound,
            "sparse_extension_layout": self.spec.extension_layout,
        })
        return details

    def prepare(self, q, k, v, lut, valid_block_num, *, layer_index, metadata):
        heads, sequence = self._validate(q, k, v, lut, valid_block_num)
        v_carrier, v_scale = self._prepare_v(v)
        self._validate_v(v, v_carrier, v_scale, sequence)
        q_int8, q_scale, k_int8, k_scale = self.qk_quantizer(q, k)
        q_blocks = (sequence + self.spec.q_tile - 1) // self.spec.q_tile
        k_blocks = (sequence + self.spec.kv_tile - 1) // self.spec.kv_tile
        if (tuple(q_int8.shape) != tuple(q.shape)
                or tuple(k_int8.shape) != tuple(k.shape)
                or q_int8.dtype != torch.int8 or k_int8.dtype != torch.int8
                or q_int8.device != q.device or k_int8.device != q.device
                or q_int8.stride(-1) != 1 or k_int8.stride(-1) != 1
                or tuple(q_scale.shape) != (1, heads, q_blocks)
                or tuple(k_scale.shape) != (1, heads, k_blocks)
                or q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32
                or q_scale.device != q.device or k_scale.device != q.device
                or not q_scale.is_contiguous() or not k_scale.is_contiguous()):
            raise SparseSageError("Q/K quantizer returned an invalid INT8 carrier")
        return PreparedSparseSage(
            q_int8=q_int8,
            k_int8=k_int8,
            v_carrier=v_carrier,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            lut=lut,
            valid_block_num=valid_block_num,
            output_dtype=q.dtype,
            output_shape=tuple(q.shape),
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            metadata=self._metadata(metadata),
            pv_threshold=torch.full(
                (heads,), 50.0, dtype=torch.float32, device=q.device
            ),
        )

    def prepare_projected(self, projected, lut, valid_block_num, *, metadata):
        from .fused_qkv import (
            sparse_fused_qkv_contract_mismatch,
            validate_prepared_fused_qkv,
        )
        validate_prepared_fused_qkv(projected)
        mismatch = sparse_fused_qkv_contract_mismatch(self.spec)
        if mismatch is not None:
            raise SparseSageError(
                "fused QKV does not match the Sparse Sage carrier contract: %s"
                % mismatch
            )
        heads, sequence = int(projected.heads), int(projected.sequence)
        self.spec.validate_lut(lut, valid_block_num, batch=1, heads=heads, sequence=sequence)
        if lut.device != projected.q_int8.device:
            raise SparseSageError("Sparse Sage LUT and fused H3 QKV devices differ")
        if not self.allow_cpu_for_tests:
            if not projected.q_int8.is_cuda:
                raise SparseSageError("fused H3 Sparse Sage requires CUDA")
            capability = tuple(torch.cuda.get_device_capability(projected.q_int8.device))
            if capability != self.spec.capability:
                raise SparseSageError(
                    "Sparse Sage resolved %s but execution device is %d.%d"
                    % (self.spec.architecture, capability[0], capability[1])
                )
        v_carrier, v_scale = self._prepare_v(projected.v)
        self._validate_v(projected.v, v_carrier, v_scale, sequence)
        details = self._metadata(metadata)
        details.update({"qkv_projection": "fused_int8", "smooth_k": bool(projected.smooth_k)})
        return PreparedSparseSage(
            q_int8=projected.q_int8,
            k_int8=projected.k_int8,
            v_carrier=v_carrier,
            q_scale=projected.q_scale,
            k_scale=projected.k_scale,
            v_scale=v_scale,
            lut=lut,
            valid_block_num=valid_block_num,
            output_dtype=projected.output_dtype,
            output_shape=(1, heads, sequence, projected.head_dim),
            layer_index=int(projected.layer_index),
            sequence=sequence,
            heads=heads,
            metadata=details,
            pv_threshold=torch.full(
                (heads,), 50.0, dtype=torch.float32,
                device=projected.q_int8.device,
            ),
        )

    def execute(self, prepared):
        try:
            if self._use_sparse_sage_op:
                output = sparse_sage_attention_op(prepared.q_int8, prepared.k_int8, prepared.v_carrier,
                    prepared.lut, prepared.valid_block_num, prepared.pv_threshold,
                    prepared.q_scale, prepared.k_scale, prepared.v_scale, prepared.output_dtype)
            else:
                output = torch.empty(
                    prepared.output_shape, dtype=prepared.output_dtype,
                    device=prepared.q_int8.device,
                )
                kernel = self.low_level_selector(prepared.q_int8)
                spec = self.spec if kernel is self.spec.kernel else replace(
                    self.spec, kernel=kernel,
                    kernel_name=getattr(kernel, "__name__", self.spec.kernel_name),
                )
                spec.dispatch(
                    prepared.q_int8, prepared.k_int8, prepared.v_carrier, output,
                    prepared.lut, prepared.valid_block_num, prepared.pv_threshold,
                    prepared.q_scale, prepared.k_scale, prepared.v_scale,
                    prepared.output_dtype,
                )
        except Exception as exc:
            raise SparseSageError(
                "Sparse Sage kernel failed: layer=%d sequence=%d heads=%d "
                "dtype=%s SpargeAttention=%s"
                % (prepared.layer_index, prepared.sequence, prepared.heads,
                   prepared.output_dtype, self.spec.version)
            ) from exc
        return output
