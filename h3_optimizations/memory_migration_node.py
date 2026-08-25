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
    FUSED_QKV_OFF,
    FUSED_QKV_PRESERVE_BF16,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
    QKV_STREAMING_AUTO,
    QKV_STREAMING_FORCED,
    QKV_STREAMING_OFF,
    MemoryRequest,
    read_plan,
)
from .status import format_memory_status

PRECISION_MODE_PRESERVE = 'Preserve precision'
PRECISION_MODE_ALLOW_FP8 = 'Allow FP8 conversion'
PRECISION_MODE_OPTIONS = (
    PRECISION_MODE_PRESERVE,
    PRECISION_MODE_ALLOW_FP8,
)

QKV_STREAMING_MODE_OFF = 'Off'
QKV_STREAMING_MODE_AUTO = 'Auto'
QKV_STREAMING_MODE_FORCED = 'Forced'
QKV_STREAMING_MODE_OPTIONS = (
    QKV_STREAMING_MODE_OFF,
    QKV_STREAMING_MODE_AUTO,
    QKV_STREAMING_MODE_FORCED,
)


def _preserve_precision_for_mode(precision_mode):
    if precision_mode == PRECISION_MODE_PRESERVE:
        return True
    if precision_mode == PRECISION_MODE_ALLOW_FP8:
        return False
    raise ValueError('unknown precision mode %r' % precision_mode)


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
    preserve_precision = _preserve_precision_for_mode(precision_mode)
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

    if streaming == QKV_STREAMING_OFF:
        qkv_request = FUSED_QKV_OFF
    elif preserve_precision:
        # Historical internal name: semantically this now means preserve the
        # checkpoint's native weight precision while preferring BF16 streaming.
        qkv_request = FUSED_QKV_PRESERVE_BF16
    else:
        # Streaming still wins. AUTO merely permits BF16/FP16 -> FP8 conversion
        # later if no compatible BF16-streamed/native provider exists.
        qkv_request = FUSED_QKV_AUTO

    mlp_request = (
        MLP_MEMORY_PRESERVE
        if preserve_precision and mlp_memory == MLP_MEMORY_AUTO
        else mlp_memory
    )

    return MemoryRequest(
        attention=attention,
        fused_qkv=qkv_request,
        mlp_memory=mlp_request,
        chunk_rows=int(chunk_rows),
        qkv_streaming=streaming,
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
                'QKV streaming defaults to Auto and prefers bounded BF16 Q/K/V '
                'chunks whenever a compatible attention consumer exists, regardless '
                'of checkpoint weight quantization. Precision mode controls whether '
                'new FP8 weight conversion is allowed only as a fallback.'
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
                        'auto uses the checkpoint-native optimized MLP path when '
                        'available. Allow FP8 conversion may convert ordinary '
                        'BF16/FP16 MLP weights to FP8 E4M3. Preserve precision keeps '
                        'floating weights floating while retaining bounded chunking. '
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
                    default=PRECISION_MODE_PRESERVE,
                    advanced=True,
                    tooltip=(
                        'Preserve precision never introduces new weight quantization. '
                        'Allow FP8 conversion permits supported BF16/FP16 weight '
                        'conversion only when a checkpoint-native or BF16-streamed '
                        'QKV path is unavailable; MLP may also use FP8 conversion.'
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
        precision_mode=PRECISION_MODE_PRESERVE,
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
