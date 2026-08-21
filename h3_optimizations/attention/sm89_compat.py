"""Compatibility discovery for SageAttention SM89 wheel variants.

Some Windows and Hugging Face kernel builds register the low-level operators in
hashed ``torch.ops`` namespaces and do not expose the wrapper functions as normal
Python module attributes. This module replaces only the resolver used by
``sage_mem_eff``; the attention implementation remains unchanged.
"""

import importlib
import sys

import torch

from . import sage_mem_eff as _target


def _append_unique(items, item, label):
    if item is None:
        return
    if not any(item is existing for existing, _ in items):
        items.append((item, label))


def _candidate_surfaces(core):
    surfaces = []
    _append_unique(surfaces, core, "sageattention.core")
    _append_unique(surfaces, getattr(core, "sm89_compile", None), "core.sm89_compile")

    public_fn = getattr(core, "sageattn_qk_int8_pv_fp8_cuda", None)
    globals_dict = getattr(public_fn, "__globals__", {}) if public_fn is not None else {}
    for name, value in globals_dict.items():
        if name.startswith("__"):
            continue
        _append_unique(surfaces, value, "public_fn.__globals__[%s]" % name)
        _append_unique(surfaces, getattr(value, "ops", None), "%s.ops" % name)

    for module_name in (
        "sageattention.sm89_compile",
        "sageattention._qattn_sm89",
        "sageattention.qattn_sm89",
        "sageattention._ops",
        "sage_attention.sm89_compile",
        "sage_attention._ops",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        _append_unique(surfaces, module, module_name)
        _append_unique(surfaces, getattr(module, "ops", None), module_name + ".ops")

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(("sageattention", "sage_attention")):
            continue
        _append_unique(surfaces, module, module_name)
        _append_unique(surfaces, getattr(module, "ops", None), module_name + ".ops")

    return surfaces


def _dispatch_table_kernels():
    """Map operator basename to callable for every registered torch op.

    Hashed kernel packages use namespaces such as
    ``_sage_attention_cuda_5568690``. Searching by basename avoids depending on
    the build-specific hash.
    """
    found = {}
    try:
        names = torch._C._dispatch_get_all_op_names()
    except Exception:
        return found

    wanted = {name for name, _, _ in _target.KERNEL_CANDIDATES}
    for full_name in names:
        if "::" not in full_name:
            continue
        namespace, basename = full_name.split("::", 1)
        if basename not in wanted:
            continue
        try:
            namespace_obj = getattr(torch.ops, namespace)
            op = getattr(namespace_obj, basename)
        except Exception:
            continue
        if callable(op):
            found.setdefault(basename, (op, "torch.ops.%s" % namespace))
    return found


def resolve_sm89_kernel(core):
    surfaces = _candidate_surfaces(core)
    dispatch = _dispatch_table_kernels()

    for kernel_name, v_scale_max, accumulation in _target.KERNEL_CANDIDATES:
        for surface, label in surfaces:
            try:
                kernel = getattr(surface, kernel_name)
            except Exception:
                continue
            if callable(kernel):
                return kernel, kernel_name, label, v_scale_max, accumulation

        if kernel_name in dispatch:
            kernel, label = dispatch[kernel_name]
            return kernel, kernel_name, label, v_scale_max, accumulation

    available = sorted(dispatch)
    expected = ", ".join(name for name, _, _ in _target.KERNEL_CANDIDATES)
    found = ", ".join(available) or "none discoverable"
    raise _target.EfficientSageError(
        "SageAttention 2.2.x has no supported SM89 kernel export. "
        "Expected one of: %s. Registered matching torch ops: %s" % (expected, found)
    )


# Replace only the resolver before any backend instance is constructed.
_target._resolve_sm89_kernel = resolve_sm89_kernel
