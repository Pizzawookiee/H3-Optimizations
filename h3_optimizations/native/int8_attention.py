"""INT8 attention over the vendored library.

The same carriers and kernels Comfy Kitchen builds, reached through the pack's
own compiled library instead of whichever comfy-kitchen pip happened to
install. Carrier layouts are deliberately identical, so a carrier produced
here is interchangeable with one produced by Kitchen and the two can be
compared byte for byte -- which is how the port is validated.

Only the transport differs. Kitchen passes tensors through nanobind and
DLPack; here it is ctypes over a plain C ABI, so there is no Python ABI
dimension to build for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .. import diagnostics
from . import loader

OUTPUT_HND = 'hnd'
OUTPUT_NHD = 'nhd'
OUTPUT_LAYOUTS = (OUTPUT_HND, OUTPUT_NHD)

Q_TILE = 128
CTA_K = 64
LARGE_CTA_K = 128
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Matches the CUDA backend's conventions; do not renumber.
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


def _pad_to(length, multiple):
    return ((length + multiple - 1) // multiple) * multiple


def select_cta_k(kernel_head_dim, kv_length, *, has_mask=False):
    """Kitchen's own tile heuristic, reproduced so carriers match."""
    if not has_mask and kernel_head_dim >= 128 and kv_length > 1024:
        return LARGE_CTA_K
    return CTA_K


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _ptr(tensor):
    return None if tensor is None else tensor.data_ptr()


@dataclass(frozen=True)
class PrequantizedInt8Attention:
    """Packed INT8 Q/K/V plus the metadata the kernels need.

    Field-for-field the same as Kitchen's carrier, including the anchor the
    quantizer centred K on -- a producer claiming to emit these has to
    reproduce the anchor choice, not only the packed bytes.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    original_head_dim: int
    input_dtype: torch.dtype
    attention_scale: float
    cta_k: int
    anchor_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class BlockSparseRoute:
    """A routed tile selection that declares its encoding and geometry.

    ``absolute`` indices are the form to reason about; ``delta`` is what the
    H3 router emits and what the kernel is built to walk. Carrying the label
    rather than assuming one is what stops a delta table being read as
    absolute, which is silent and produces a plausible wrong answer.
    """

    indices: torch.Tensor
    counts: torch.Tensor
    q_tile: int
    kv_tile: int
    encoding: str = 'absolute'

    def __post_init__(self):
        if self.encoding not in ('absolute', 'delta'):
            raise ValueError(
                "encoding must be 'absolute' or 'delta', got %r" % self.encoding
            )

    def _live(self):
        index = torch.arange(self.indices.shape[-1], device=self.indices.device)
        return index < self.counts.to(torch.int64).unsqueeze(-1)

    def to_absolute(self):
        if self.encoding == 'absolute':
            return self
        positions = self.indices.to(torch.int64).cumsum(dim=-1)
        indices = torch.where(
            self._live(), positions, torch.zeros_like(positions)
        ).to(torch.int32)
        return replace(self, indices=indices.contiguous(), encoding='absolute')

    def to_delta(self):
        if self.encoding == 'delta':
            return self
        tiles = self.indices.to(torch.int64)
        previous = torch.cat(
            (torch.zeros_like(tiles[..., :1]), tiles[..., :-1]), dim=-1
        )
        steps = torch.where(
            torch.arange(tiles.shape[-1], device=tiles.device) == 0,
            tiles,
            tiles - previous,
        )
        indices = torch.where(
            self._live(), steps, torch.zeros_like(steps)
        ).to(torch.int32)
        return replace(self, indices=indices.contiguous(), encoding='delta')

    def for_kernel(self):
        """The route in whatever encoding the compiled kernel walks."""
        return (
            self.to_delta()
            if loader.route_encoding() == 'delta'
            else self.to_absolute()
        )


def _check_quantize_qk(*arguments):
    loader.check(loader.load().h3_int8_quantize_qk(*arguments), 'quantize_qk')


def _check_quantize_q(*arguments):
    loader.check(loader.load().h3_int8_quantize_q(*arguments), 'quantize_q')


def _check_quantize_v(*arguments):
    loader.check(loader.load().h3_int8_quantize_v(*arguments), 'quantize_v')


def _check_sparse_attention(*arguments):
    loader.check(
        loader.load().h3_int8_sparse_attention(*arguments), 'sparse attention'
    )


def _kernel_head_dim(head_dim):
    if head_dim <= 64:
        return 64
    if head_dim <= 128:
        return 128
    return 256


def _validate(q, k, v):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('q, k and v must be [batch, heads, sequence, head_dim]')
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError('q, k and v must be float32, float16 or bfloat16')
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError('q, k and v must share a dtype')
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError('q, k and v must be CUDA tensors')
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError('the head dimension of q, k and v must be contiguous')


def prequantize_int8_attention(q, k, v, *, scale=None, cta_k=None):
    """Quantize Q/K/V into the carriers the attention kernels consume."""
    library = loader.load()
    _validate(q, k, v)
    if cta_k is not None and cta_k not in (CTA_K, LARGE_CTA_K):
        raise ValueError('cta_k must be %d or %d' % (CTA_K, LARGE_CTA_K))

    original_head_dim = q.shape[-1]
    input_dtype = q.dtype
    kernel_head_dim = _kernel_head_dim(original_head_dim)
    if kernel_head_dim != original_head_dim:
        padding = (0, kernel_head_dim - original_head_dim)
        q = torch.nn.functional.pad(q, padding)
        k = torch.nn.functional.pad(k, padding)
        v = torch.nn.functional.pad(v, padding)

    attention_scale = (
        original_head_dim ** -0.5 if scale is None else float(scale)
    )
    batch, q_heads, q_length, _ = q.shape
    _, kv_heads, kv_length, _ = k.shape
    if cta_k is None:
        cta_k = select_cta_k(kernel_head_dim, kv_length)
    padded_k_length = _pad_to(kv_length, cta_k)

    device = q.device
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=device)
    q_scales_per_tile = 64 if kernel_head_dim == 256 else 32
    q_scale = torch.empty(
        batch, q_heads, ((q_length + Q_TILE - 1) // Q_TILE) * q_scales_per_tile,
        dtype=torch.float32, device=device,
    )
    k_scale = torch.empty(
        batch, kv_heads, ((kv_length + cta_k - 1) // cta_k) * 4,
        dtype=torch.float32, device=device,
    )
    v_int8 = torch.empty(
        batch * kv_heads * kernel_head_dim, padded_k_length,
        dtype=torch.int8, device=device,
    )
    v_scale = torch.empty(
        batch * kv_heads * kernel_head_dim, dtype=torch.float32, device=device
    )
    anchor_indices = torch.empty(
        batch, kv_heads, dtype=torch.int32, device=device
    )

    dtype_code = _DTYPE_TO_CODE[input_dtype]
    warp_q = 16 if kernel_head_dim == 256 else 32
    with diagnostics.stage('qk_carrier_pack'):
        _check_quantize_qk(
            _ptr(q), _ptr(q_int8), _ptr(q_scale),
            _ptr(k), _ptr(k_int8), _ptr(k_scale),
            batch, q_heads, q_length, kv_heads, kv_length, kernel_head_dim,
            Q_TILE, warp_q, cta_k, cta_k,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            dtype_code, _ptr(anchor_indices), _stream(),
        )
    with diagnostics.stage('v_carrier_pack'):
        _check_quantize_v(
            _ptr(v), _ptr(v_int8), _ptr(v_scale),
            batch, kv_heads, kv_length, kernel_head_dim, padded_k_length,
            v.stride(0), v.stride(1), v.stride(2),
            dtype_code, _stream(),
        )
    return PrequantizedInt8Attention(
        q=q_int8, k=k_int8, v=v_int8,
        q_scale=q_scale, k_scale=k_scale, v_scale=v_scale,
        original_head_dim=original_head_dim,
        input_dtype=input_dtype,
        attention_scale=attention_scale,
        cta_k=cta_k,
        anchor_indices=anchor_indices,
    )


def quantize_int8_attention_q(q, *, full_k_length):
    '''Quantize one local H3 query slab without constructing a K carrier.'''
    if q.ndim != 4:
        raise ValueError('q must be [batch, heads, sequence, head_dim]')
    if q.dtype not in _SUPPORTED_DTYPES:
        raise TypeError('q must be float32, float16 or bfloat16')
    if not q.is_cuda:
        raise ValueError('q must be a CUDA tensor')
    if q.stride(-1) != 1:
        raise ValueError('the Q head dimension must be contiguous')
    if int(q.shape[-1]) != 128:
        raise ValueError(
            'streamed native Q quantization requires H3 head_dim 128'
        )

    batch, heads, q_length, head_dim = q.shape
    full_k_length = int(full_k_length)
    if full_k_length <= 0:
        raise ValueError('full_k_length must be positive')

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    q_scale = torch.empty(
        batch,
        heads,
        ((q_length + Q_TILE - 1) // Q_TILE) * 32,
        dtype=torch.float32,
        device=q.device,
    )
    with diagnostics.stage('q_carrier_pack'):
        _check_quantize_q(
            _ptr(q), _ptr(q_int8), _ptr(q_scale),
            batch, heads, q_length, head_dim, full_k_length,
            q.stride(0), q.stride(1), q.stride(2),
            _DTYPE_TO_CODE[q.dtype], _stream(),
        )
    return q_int8, q_scale


def _attention_geometry(quantized, output_layout=OUTPUT_HND):
    """Allocate the output and describe every tensor to the kernel by stride.

    The kernel addresses O purely through ``stride_bz_o/stride_seq_o/
    stride_h_o``, so the two layouts differ only in the numbers passed here --
    no kernel branch, no second code path, and the same 16-byte store
    granularity within a head either way.

    Both layouts return a ``[batch, heads, sequence, head_dim]`` tensor. Only
    the physical storage differs, and that is the whole point: every caller
    ends up wanting ``[sequence, heads * head_dim]``, which it reaches with
    ``transpose(1, 2).reshape(...)``. Out of head-major storage those two
    dimensions cannot merge, so the reshape materializes a second
    full-sequence buffer -- 738 MiB at the production shape. Out of
    sequence-major storage the transpose lands back on contiguous memory and
    the reshape is a view. The returned tensor is a permuted view either way,
    so nothing downstream has to know which it got.
    """
    if output_layout not in OUTPUT_LAYOUTS:
        raise ValueError(
            'output_layout must be one of %s, got %r'
            % (', '.join(OUTPUT_LAYOUTS), output_layout)
        )
    batch, q_heads, q_length, kernel_head_dim = quantized.q.shape
    kv_heads, kv_length = quantized.k.shape[1], quantized.k.shape[2]
    padded_k_length = _pad_to(kv_length, quantized.cta_k)
    output_dtype = (
        torch.bfloat16
        if quantized.input_dtype == torch.float32
        else quantized.input_dtype
    )
    if output_layout == OUTPUT_NHD:
        storage = torch.empty(
            batch, q_length, q_heads, kernel_head_dim,
            dtype=output_dtype, device=quantized.q.device,
        )
        # Logical BHND over sequence-major storage. The kernel is told the
        # storage strides; the caller sees the shape it always saw.
        output = storage.permute(0, 2, 1, 3)
        output_strides = (
            q_length * q_heads * kernel_head_dim,
            q_heads * kernel_head_dim,
            kernel_head_dim,
        )
    else:
        output = torch.empty(
            batch, q_heads, q_length, kernel_head_dim,
            dtype=output_dtype, device=quantized.q.device,
        )
        output_strides = (
            q_heads * q_length * kernel_head_dim,
            kernel_head_dim,
            q_length * kernel_head_dim,
        )
    strides = (
        q_heads * q_length * kernel_head_dim, kernel_head_dim,
        q_length * kernel_head_dim,
        kv_heads * kv_length * kernel_head_dim, kernel_head_dim,
        kv_length * kernel_head_dim,
        kv_heads * kernel_head_dim * padded_k_length,
        kernel_head_dim * padded_k_length, padded_k_length,
    ) + output_strides
    return output, output_dtype, (
        batch, q_length, kv_length, q_heads, kv_heads, kernel_head_dim
    ), strides


def _finish(quantized, output):
    output = output[..., : quantized.original_head_dim]
    return output.float() if quantized.input_dtype == torch.float32 else output


def int8_attention_from_prequantized(quantized, *, output_layout=OUTPUT_HND):
    """Dense INT8 attention over every KV tile."""
    library = loader.load()
    output, output_dtype, geometry, strides = _attention_geometry(
        quantized, output_layout
    )
    loader.check(
        library.h3_int8_dense_attention(
            _ptr(quantized.q), _ptr(quantized.k), _ptr(quantized.v),
            _ptr(output), _ptr(quantized.q_scale), _ptr(quantized.k_scale),
            _ptr(quantized.v_scale), quantized.cta_k, *geometry, *strides,
            quantized.attention_scale, _DTYPE_TO_CODE[output_dtype], _stream(),
        ),
        'dense attention',
    )
    return _finish(quantized, output)


def block_sparse_int8_attention_from_prequantized(
    quantized, route, *, output_layout=OUTPUT_HND
):
    """INT8 attention over the KV tiles the route selects."""
    if not isinstance(route, BlockSparseRoute):
        raise TypeError(
            'route must be a BlockSparseRoute so its encoding and tile '
            'geometry are explicit, got %s' % type(route).__name__
        )
    if route.kv_tile != quantized.cta_k:
        raise ValueError(
            'the route was built for KV tile %d but the carriers are packed '
            'for %d; walking one at the other width attends to the wrong keys'
            % (route.kv_tile, quantized.cta_k)
        )
    if route.q_tile != Q_TILE:
        raise ValueError('block-sparse attention is fixed to q tile %d' % Q_TILE)

    library = loader.load()
    kernel_route = route.for_kernel()
    if not kernel_route.indices.is_contiguous() or not kernel_route.counts.is_contiguous():
        raise ValueError('route indices and counts must be contiguous')
    if kernel_route.indices.dtype != torch.int32 or kernel_route.counts.dtype != torch.int32:
        raise TypeError('route indices and counts must be int32')

    output, output_dtype, geometry, strides = _attention_geometry(
        quantized, output_layout
    )
    kv_tiles = kernel_route.indices.shape[-1]
    with diagnostics.stage('sparse_attention_kernel'):
        _check_sparse_attention(
            _ptr(quantized.q), _ptr(quantized.k), _ptr(quantized.v),
            _ptr(output), _ptr(quantized.q_scale), _ptr(quantized.k_scale),
            _ptr(quantized.v_scale), _ptr(kernel_route.indices),
            _ptr(kernel_route.counts), kv_tiles, quantized.cta_k,
            *geometry, *strides,
            quantized.attention_scale, _DTYPE_TO_CODE[output_dtype],
            _stream(),
        )
    return _finish(quantized, output)


def int8_attention_is_available(device=None):
    """Whether the vendored kernels can run here.

    Mirrors comfy_kitchen's predicate so callers can hold either module
    without branching on which one they got.
    """
    if not torch.cuda.is_available():
        return False
    if not loader.is_available():
        return False
    capability = torch.cuda.get_device_capability(device)
    if tuple(capability) < (8, 0):
        return False
    from . import selftest

    return selftest.check(device)
