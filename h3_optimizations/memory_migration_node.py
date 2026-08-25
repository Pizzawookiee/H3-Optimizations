'''Workflow-compatible public H3 Memory Optimization node.

Legacy widgets remain in their original serialized positions so positional
ComfyUI workflows continue to deserialize correctly. They are hidden and
ignored by execution. Appended authoritative controls own current behavior.
'''

from comfy_api.latest import io, ui

from .apply_policy import apply_plan
from .dense_resolver import has_explicit_dense_attention
from .nodes import DEFAULT_CHUNK_ROWS, NODE_CATEGORY
from .plan import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_FORCE_BF16,
    FUSED_QKV_FORCE_QUANT,
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_BF16,
    MLP_MEMORY_FORCE_QUANT,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
    QKV_STREAMING_AUTO,
    QKV_STREAMING_FORCED,
    QKV_STREAMING_OFF,
    MemoryRequest,
    read_plan,
)
from .status import format_memory_status

PRECISION_MODE_AUTO = 'Auto'
PRECISION_MODE_BF16 = 'BF16'
PRECISION_MODE_PRESERVE_NATIVE = 'Preserve native'
PRECISION_MODE_FORCE_QUANT = 'Force quant'
PRECISION_MODE_OPTIONS = (
    PRECISION_MODE_AUTO,
    PRECISION_MODE_BF16,
    PRECISION_MODE_PRESERVE_NATIVE,
    PRECISION_MODE_FORCE_QUANT,
)
PRECISION_MODE_PRESERVE = 'Preserve precision'
PRECISION_MODE_ALLOW_FP8 = 'Allow FP8 conversion'
_PRECISION_MODE_COMPATIBILITY = {
    PRECISION_MODE_PRESERVE: PRECISION_MODE_PRESERVE_NATIVE,
    PRECISION_MODE_ALLOW_FP8: PRECISION_MODE_AUTO,
}

QKV_STREAMING_MODE_OFF = 'Off'
QKV_STREAMING_MODE_AUTO = 'Auto'
QKV_STREAMING_MODE_FORCED = 'Forced'
QKV_STREAMING_MODE_OPTIONS = (
    QKV_STREAMING_MODE_OFF,
    QKV_STREAMING_MODE_AUTO,
    QKV_STREAMING_MODE_FORCED,
)


def _normalize_precision_mode(precision_mode):
    mode = _PRECISION_MODE_COMPATIBILITY.get(precision_mode, precision_mode)
    if mode not in PRECISION_MODE_OPTIONS:
        raise ValueError('unknown precision mode %r' % precision_mode)
    return mode


def _qkv_streaming_request(mode):
    if mode == QKV_STREAMING_MODE_OFF:
        return QKV_STREAMING_OFF
    if mode == QKV_STREAMING_MODE_AUTO:
        return QKV_STREAMING_AUTO
    if mode == QKV_STREAMING_MODE_FORCED:
        return QKV_STREAMING_FORCED
    raise ValueError('unknown QKV streaming mode %r' % mode)


def _memory_request_for_modes(
    *,
    fused_qkv,
    mlp_memory,
    chunk_rows,
    precision_mode,
    qkv_streaming_mode,
    explicit_attention_selected=False,
):
    # fused_qkv is a serialized compatibility tombstone. QKV streaming is the
    # authoritative public policy now; keeping the argument preserves old
    # positional workflows without preserving its old conflicting semantics.
    del fused_qkv
    precision_mode = _normalize_precision_mode(precision_mode)
    streaming = _qkv_streaming_request(qkv_streaming_mode)

    # Off preserves the current attention backend. Auto claims unselected dense
    # attention only to obtain a real streamed producer and yields to an
    # explicit selector. Forced explicitly authorizes the Kitchen dense path.
    # An H3 Sparse request remains separately authoritative through the shared
    # order-independent optimization plan.
    attention = (
        ATTENTION_EXISTING
        if streaming == QKV_STREAMING_OFF
        or (streaming == QKV_STREAMING_AUTO and explicit_attention_selected)
        else ATTENTION_AUTO
    )

    qkv_requests = {
        PRECISION_MODE_AUTO: FUSED_QKV_AUTO,
        PRECISION_MODE_BF16: FUSED_QKV_FORCE_BF16,
        PRECISION_MODE_PRESERVE_NATIVE: FUSED_QKV_PRESERVE_BF16,
        PRECISION_MODE_FORCE_QUANT: FUSED_QKV_FORCE_QUANT,
    }
    qkv_request = (
        FUSED_QKV_OFF
        if streaming == QKV_STREAMING_OFF
        else qkv_requests[precision_mode]
    )

    mlp_requests = {
        PRECISION_MODE_AUTO: MLP_MEMORY_AUTO,
        PRECISION_MODE_BF16: MLP_MEMORY_BF16,
        PRECISION_MODE_PRESERVE_NATIVE: MLP_MEMORY_PRESERVE,
        PRECISION_MODE_FORCE_QUANT: MLP_MEMORY_FORCE_QUANT,
    }
    mlp_request = (
        mlp_requests[precision_mode]
        if mlp_memory == MLP_MEMORY_AUTO
        else mlp_memory
    )

    return MemoryRequest(
        attention=attention,
        fused_qkv=qkv_request,
        mlp_memory=mlp_request,
        chunk_rows=int(chunk_rows),
        qkv_streaming=streaming,
        mlp_strict=precision_mode in (
            PRECISION_MODE_BF16,
            PRECISION_MODE_FORCE_QUANT,
        ),
    )


class H3MemoryOptimization(io.ComfyNode):
    '''Chunked QKV and bounded MLP execution with workflow-safe precision policy.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MemoryOptimization',
            display_name='H3 Memory Optimization',
            category=NODE_CATEGORY,
            description=(
                'Production memory and execution optimizations for MiniMax H3. '
                'QKV streaming defaults to Auto and prefers bounded Q/K/V chunks '
                'whenever a compatible attention consumer exists. Precision mode '
                'selects automatic, BF16, checkpoint-native, or forced quantized '
                'weight execution.'
            ),
            search_aliases=[
                'H3 VRAM',
                'H3 memory',
                'H3 fused QKV',
                'H3 streamed QKV',
                'H3 chunked MLP',
                'H3 preserve precision',
                'H3 non quantized memory',
                'MiniMax memory optimizer',
                'Sage optimizer',
            ],
            inputs=[
                io.Model.Input('model'),
                # Positional compatibility tombstone from <=0.2.5. Do not remove,
                # reorder, or mark non-serializable: old workflows store this slot.
                io.Combo.Input(
                    'fused_qkv',
                    display_name='Legacy QKV projection optimization',
                    options=[FUSED_QKV_AUTO, FUSED_QKV_OFF],
                    default=FUSED_QKV_AUTO,
                    advanced=True,
                    extra_dict={
                        'hidden': True,
                        'tooltip': (
                            'Legacy serialized workflow slot. The value is ignored; '
                            'QKV streaming is authoritative.'
                        ),
                    },
                ),
                io.Combo.Input(
                    'mlp_memory',
                    display_name='MLP memory optimization',
                    options=[MLP_MEMORY_AUTO, MLP_MEMORY_OFF],
                    default=MLP_MEMORY_AUTO,
                    tooltip=(
                        'auto applies the selected precision policy while retaining '
                        'bounded MLP chunking. '
                        'Explicit off remains off.'
                    ),
                ),
                io.Int.Input(
                    'chunk_rows',
                    display_name='Activation chunk rows',
                    default=DEFAULT_CHUNK_ROWS,
                    min=MIN_CHUNK_ROWS,
                    max=MAX_CHUNK_ROWS,
                    step=256,
                    advanced=True,
                    tooltip=(
                        'Maximum token rows processed by one MLP or FinalLayer '
                        'chunk. Larger chunks may be faster but use more activation '
                        'memory.'
                    ),
                ),
                # Original preserve_precision slot. Keep serialized and hidden.
                io.Boolean.Input(
                    'preserve_precision',
                    default=True,
                    advanced=True,
                    extra_dict={
                        'hidden': True,
                        'tooltip': (
                            'Legacy serialized workflow slot. The value is ignored; '
                            'Precision mode is authoritative.'
                        ),
                    },
                ),
                io.Combo.Input(
                    'precision_mode',
                    display_name='Precision mode',
                    options=list(PRECISION_MODE_OPTIONS),
                    default=PRECISION_MODE_AUTO,
                    advanced=True,
                    tooltip=(
                        'Auto selects the best compatible native path and may use FP8 '
                        'conversion as a fallback. BF16 materializes supported weights '
                        'as BF16. Preserve native never introduces a new conversion. '
                        'Force quant keeps supported quantized checkpoints native and '
                        'requires floating weights to use FP8 E4M3.'
                    ),
                ),
                io.Combo.Input(
                    'qkv_streaming_mode',
                    display_name='QKV streaming',
                    options=list(QKV_STREAMING_MODE_OPTIONS),
                    default=QKV_STREAMING_MODE_AUTO,
                    advanced=True,
                    tooltip=(
                        'Off disables streamed QKV and preserves existing attention. '
                        'Auto prefers BF16 Q/K/V chunk streaming and only claims dense '
                        'Kitchen when a compatible streamed producer is available; it '
                        'preserves explicit attention selectors. Forced explicitly '
                        'allows this node to replace dense attention with full-density '
                        'Kitchen. H3 Sparse Attention remains authoritative.'
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        fused_qkv=FUSED_QKV_AUTO,
        mlp_memory=MLP_MEMORY_AUTO,
        chunk_rows=DEFAULT_CHUNK_ROWS,
        preserve_precision=True,
        precision_mode=PRECISION_MODE_AUTO,
        qkv_streaming_mode=QKV_STREAMING_MODE_AUTO,
    ):
        del preserve_precision
        plan = read_plan(model).with_memory(
            _memory_request_for_modes(
                fused_qkv=fused_qkv,
                mlp_memory=mlp_memory,
                chunk_rows=chunk_rows,
                precision_mode=precision_mode,
                qkv_streaming_mode=qkv_streaming_mode,
                explicit_attention_selected=has_explicit_dense_attention(model),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_memory_status(patched)),
        )

    @classmethod
    def validate_inputs(cls, precision_mode):
        try:
            _normalize_precision_mode(precision_mode)
        except ValueError as exc:
            return str(exc)
        return True
