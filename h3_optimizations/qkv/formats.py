"""Format-neutral inspection for H3 QKV and MLP linear weights."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearWeightFormat:
    layout_name: str | None
    quantized: bool
    storage_dtype: str
    shape: tuple[int, ...]
    transposed: bool
    convrot: bool
    convrot_group_size: int | None
    has_bias: bool
    weight_type: str

    @property
    def tensorwise_int8(self):
        return self.layout_name == "TensorWiseINT8Layout"

    @property
    def convrot_int8_256(self):
        return (
            self.quantized
            and self.tensorwise_int8
            and not self.transposed
            and self.convrot
            and self.convrot_group_size == 256
            and not self.has_bias
        )

    @property
    def label(self):
        if self.layout_name:
            suffix = ""
            if self.convrot:
                suffix = "+convrot%d" % int(self.convrot_group_size or 0)
            return "%s%s" % (self.layout_name, suffix)
        return "%s:%s" % (self.weight_type, self.storage_dtype)


@dataclass(frozen=True)
class H3LinearInventory:
    qkv: tuple[LinearWeightFormat, ...]
    fc1: tuple[LinearWeightFormat, ...]
    fc2: tuple[LinearWeightFormat, ...]

    @property
    def qkv_convrot_int8_256(self):
        return bool(self.qkv) and all(item.convrot_int8_256 for item in self.qkv)

    @property
    def mlp_convrot_int8_256(self):
        return (
            bool(self.fc1)
            and len(self.fc1) == len(self.fc2)
            and all(item.convrot_int8_256 for item in self.fc1)
            and all(item.convrot_int8_256 for item in self.fc2)
        )

    def labels(self, name):
        return tuple(item.label for item in getattr(self, name))

    def homogeneous(self, name):
        values = self.labels(name)
        return bool(values) and len(set(values)) == 1


def _is_quantized(weight):
    if getattr(weight, "_layout_cls", None) is not None:
        return True
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return False
    return isinstance(weight, QuantizedTensor)


def _storage_dtype(weight):
    dtype = getattr(weight, "dtype", None)
    if dtype is not None:
        return str(dtype)
    value = getattr(weight, "_value", None)
    dtype = getattr(value, "dtype", None)
    return "unknown" if dtype is None else str(dtype)


def describe_weight(weight, *, bias=None):
    params = getattr(weight, "_params", None)
    group_size = getattr(params, "convrot_groupsize", None)
    try:
        group_size = None if group_size is None else int(group_size)
    except Exception:
        group_size = None
    shape = tuple(int(value) for value in getattr(weight, "shape", ()))
    return LinearWeightFormat(
        layout_name=getattr(weight, "_layout_cls", None),
        quantized=_is_quantized(weight),
        storage_dtype=_storage_dtype(weight),
        shape=shape,
        transposed=bool(getattr(params, "transposed", False)),
        convrot=bool(getattr(params, "convrot", False)),
        convrot_group_size=group_size,
        has_bias=bias is not None,
        weight_type=type(weight).__name__,
    )


def describe_linear(module):
    return describe_weight(
        getattr(module, "weight", None),
        bias=getattr(module, "bias", None),
    )


def inspect_h3_linears(blocks):
    qkv = []
    fc1 = []
    fc2 = []
    for index, block in enumerate(tuple(blocks)):
        attn = getattr(block, "attn", None)
        mlp = getattr(block, "mlp", None)
        if attn is None or mlp is None:
            raise ValueError("H3 block %d is missing attention or MLP" % index)
        qkv_proj = getattr(attn, "qkv_proj", None)
        first = getattr(mlp, "fc1", None)
        second = getattr(mlp, "fc2", None)
        if qkv_proj is None or first is None or second is None:
            raise ValueError("H3 block %d is missing qkv_proj/fc1/fc2" % index)
        qkv.append(describe_linear(qkv_proj))
        fc1.append(describe_linear(first))
        fc2.append(describe_linear(second))
    return H3LinearInventory(tuple(qkv), tuple(fc1), tuple(fc2))


_WEIGHT_FORMAT_MESSAGES = (
    "does not support projection bias",
    "requires a quantized projection weight",
    "requires TensorWise INT8 weights",
    "does not support transposed weights",
    "requires ConvRot-256 weights",
)


def is_fused_weight_format_error(exc):
    text = str(exc)
    return any(fragment in text for fragment in _WEIGHT_FORMAT_MESSAGES)
