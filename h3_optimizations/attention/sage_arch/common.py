"""Shared prepared-QKV Sage architecture infrastructure."""

from dataclasses import dataclass
import importlib
import importlib.metadata
import logging
import sys

import torch

from .. import stats
from ..sage_mem_eff import EfficientSageError, max_linear_offset

SUPPORTED_SAGE_PREFIXES = ("2.2.",)
SIGNED_OFFSET_LIMIT = (1 << 31) - 1
LOG_PREFIX = "[H3 attention]"


@dataclass(frozen=True)
class KernelBinding:
    fn: object
    name: str
    source: str


@dataclass
class PreparedArchitecture:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_source: torch.Tensor
    output_dtype: torch.dtype
    layer_index: int
    sequence: int
    heads: int
    head_dim: int
    softmax_scale: float


def load_core():
    try:
        version = importlib.metadata.version("sageattention")
        import sageattention.core as core
    except Exception as exc:
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "prepared Sage backends require SageAttention 2.2.x"
        ) from exc

    if not version.startswith(SUPPORTED_SAGE_PREFIXES):
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "prepared Sage backends were validated against SageAttention "
            "2.2.x; installed version is %s" % version
        )
    return version, core


def _append_unique(items, item, label):
    if item is None:
        return
    if not any(item is existing for existing, _ in items):
        items.append((item, label))


def _candidate_surfaces(core, family, public_names):
    surfaces = []
    _append_unique(surfaces, core, "sageattention.core")
    _append_unique(
        surfaces,
        getattr(core, "%s_compile" % family, None),
        "core.%s_compile" % family,
    )

    for public_name in public_names:
        public_fn = getattr(core, public_name, None)
        globals_dict = (
            getattr(public_fn, "__globals__", {})
            if public_fn is not None
            else {}
        )
        for name, value in globals_dict.items():
            if name.startswith("__"):
                continue
            label = "%s.__globals__[%s]" % (public_name, name)
            _append_unique(surfaces, value, label)
            _append_unique(
                surfaces,
                getattr(value, "ops", None),
                label + ".ops",
            )

    module_names = (
        "sageattention.%s_compile" % family,
        "sageattention._qattn_%s" % family,
        "sageattention.qattn_%s" % family,
        "sageattention._ops",
        "sage_attention.%s_compile" % family,
        "sage_attention._ops",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        _append_unique(surfaces, module, module_name)
        _append_unique(
            surfaces,
            getattr(module, "ops", None),
            module_name + ".ops",
        )

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(
            ("sageattention", "sage_attention")
        ):
            continue
        _append_unique(surfaces, module, module_name)
        _append_unique(
            surfaces,
            getattr(module, "ops", None),
            module_name + ".ops",
        )
    return surfaces


def _registered_ops(kernel_names):
    wanted = set(kernel_names)
    found = {}
    try:
        names = torch._C._dispatch_get_all_op_names()
    except Exception:
        return found

    for full_name in names:
        if "::" not in full_name:
            continue
        namespace, basename = full_name.split("::", 1)
        if basename not in wanted:
            continue
        try:
            op = getattr(
                getattr(torch.ops, namespace),
                basename,
            )
        except Exception:
            continue
        if callable(op):
            found.setdefault(
                basename,
                KernelBinding(
                    op,
                    basename,
                    "torch.ops.%s" % namespace,
                ),
            )
    return found


def resolve_kernel(core, family, kernel_names, public_names):
    surfaces = _candidate_surfaces(
        core,
        family,
        public_names,
    )
    dispatch = _registered_ops(kernel_names)
    for kernel_name in kernel_names:
        for surface, label in surfaces:
            try:
                kernel = getattr(surface, kernel_name)
            except Exception:
                continue
            if callable(kernel):
                return KernelBinding(
                    kernel,
                    kernel_name,
                    label,
                )
        if kernel_name in dispatch:
            return dispatch[kernel_name]

    raise EfficientSageError(
        "SageAttention 2.2.x has no callable %s kernel export; "
        "expected %s"
        % (family.upper(), ", ".join(kernel_names))
    )


def independent_contiguous(tensor):
    result = tensor.contiguous()
    if (
        result.untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    ):
        result = tensor.clone(
            memory_format=torch.contiguous_format
        )
    return result


def guard_signed_offsets(tensor):
    """Copy before stock Triton/CUDA quantizers can wrap int32 offsets."""
    if max_linear_offset(tensor) <= SIGNED_OFFSET_LIMIT:
        return tensor

    stats.increment("qk_guard_copies")
    contiguous = tensor.contiguous()
    if max_linear_offset(contiguous) > SIGNED_OFFSET_LIMIT:
        raise EfficientSageError(
            "prepared Sage cannot safely quantize a tensor with "
            "more than 2**31 addressable elements"
        )
    return contiguous


class ArchitectureBackend:
    name = "sage_mem_eff_arch"
    capabilities = frozenset()

    def __init__(self, allow_cpu_for_tests=False):
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self._logged = False
        stats.increment("configured")

    def validate(self, q, k, v):
        if q.shape != k.shape or q.shape != v.shape:
            raise EfficientSageError(
                "%s requires equal Q/K/V shapes; got %s %s %s"
                % (
                    self.name,
                    tuple(q.shape),
                    tuple(k.shape),
                    tuple(v.shape),
                )
            )
        if q.ndim != 4:
            raise EfficientSageError(
                "%s expects HND rank-4 tensors" % self.name
            )
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise EfficientSageError(
                "%s expects H3 attention batch 1; got %d"
                % (self.name, batch)
            )
        if head_dim != 128:
            raise EfficientSageError(
                "%s supports H3 head_dim 128; got %d"
                % (self.name, head_dim)
            )
        if q.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise EfficientSageError(
                "%s requires fp16 or bf16; got %s"
                % (self.name, q.dtype)
            )
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise EfficientSageError(
                "%s received differing Q/K/V dtypes"
                % self.name
            )
        if q.device != k.device or q.device != v.device:
            raise EfficientSageError(
                "%s received differing Q/K/V devices"
                % self.name
            )
        if (
            q.stride(-1) != 1
            or k.stride(-1) != 1
            or v.stride(-1) != 1
        ):
            raise EfficientSageError(
                "%s requires a contiguous head dimension"
                % self.name
            )

        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise EfficientSageError(
                    "%s requires CUDA" % self.name
                )
            capability = tuple(
                torch.cuda.get_device_capability(q.device)
            )
            if capability not in self.capabilities:
                supported = ", ".join(
                    "SM%d%d" % item
                    for item in sorted(self.capabilities)
                )
                raise EfficientSageError(
                    "%s supports %s; device capability is %d.%d"
                    % (
                        self.name,
                        supported,
                        capability[0],
                        capability[1],
                    )
                )
        return batch, heads, sequence, head_dim

    def prepared(
        self,
        q,
        q_int8,
        q_scale,
        k_int8,
        k_scale,
        v_source,
        *,
        layer_index,
        heads,
        sequence,
        head_dim,
    ):
        stats.observe_sequence(sequence)
        return PreparedArchitecture(
            q_int8=q_int8,
            q_scale=q_scale,
            k_int8=k_int8,
            k_scale=k_scale,
            v_source=v_source,
            output_dtype=q.dtype,
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            head_dim=int(head_dim),
            softmax_scale=head_dim**-0.5,
        )

    def log_once(self, version, detail):
        if self._logged:
            return
        logging.info(
            "%s %s active: SageAttention %s, %s",
            LOG_PREFIX,
            self.name,
            version,
            detail,
        )
        self._logged = True

    def kernel_error(self, prepared, kernel_name, exc):
        stats.increment("kernel_errors")
        device = prepared.q_int8.device
        gpu = (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        )
        raise EfficientSageError(
            "%s kernel failed: layer=%d sequence=%d "
            "heads=%d head_dim=%d dtype=%s device=%s "
            "kernel=%s"
            % (
                self.name,
                prepared.layer_index,
                prepared.sequence,
                prepared.heads,
                prepared.head_dim,
                prepared.output_dtype,
                gpu,
                kernel_name,
            )
        ) from exc
