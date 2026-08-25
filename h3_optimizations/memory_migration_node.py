'''Workflow-compatible public H3 Memory Optimization node.

The legacy preserve_precision boolean must remain in its original serialized
widget position so old positional ComfyUI workflows continue to deserialize
correctly. It is hidden from the UI and intentionally ignored by execution.
New authoritative controls are appended after every legacy widget so old
workflows adopt their defaults on first load.
'''

from comfy_api.latest import io, ui

from .apply import apply_plan
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
    preserve_precision = _preserve_precision_for_mode(precision_mode)
    streaming = _qkv_streaming_request(qkv_streaming_mode)

    # Off always preserves the existing attention choice. Auto claims the
    # unselected/default attention path, but yields to an explicit upstream
    # selector. Forced explicitly authorizes replacing the dense attention
    # choice with full-density Kitchen. An H3 Sparse request remains separately
    # authoritative through the shared order-independent plan.
    attention = (
        ATTENTION_EXISTING
        if streaming == QKV_STREAMING_OFF
        or (streaming == QKV_STREAMING_AUTO and explicit_attention_selected)
        else ATTENTION_AUTO
    )

    if preserve_precision:
        qkv_request = (
            FUSED_QKV_OFF
            if streaming == QKV_STREAMING_OFF
            else FUSED_QKV_PRESERVE_BF16
        )
        mlp_request = (
            MLP_MEMORY_PRESERVE
            if mlp_memory == MLP_MEMORY_AUTO
            else mlp_memory
        )
    else:
        qkv_request = FUSED_QKV_OFF if streaming == QKV_STREAMING_OFF else fused_qkv
        mlp_request = mlp_memory

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
                'Precision mode controls whether new weight quantization is allowed. '
                'QKV streaming defaults to Auto: when attention is unclaimed it uses '
                'the full-density Kitchen streaming path; an explicit attention '
                'selection is preserved. Forced overrides the dense attention choice.'
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
                io.Combo.Input(
                    'fused_qkv',
                    display_name='QKV projection optimization',
                    options=[FUSED_QKV_AUTO, FUSED_QKV_OFF],
                    default=FUSED_QKV_AUTO,
                    tooltip=(
                        'auto uses compatible chunked QKV projection providers. '
                        'ConvRot INT8 keeps its specialized path; checkpoint-native '
                        'FP8 uses held FP8 projection. With Allow FP8 conversion, '
                        'BF16/FP16 may be converted to FP8 E4M3. Unsupported '
                        'quantized formats use standard Comfy QKV. off always uses '
                        'standard H3 QKV.'
                    ),
                ),
                io.Combo.Input(
                    'mlp_memory',
                    display_name='MLP memory optimization',
                    options=[MLP_MEMORY_AUTO, MLP_MEMORY_OFF],
                    default=MLP_MEMORY_AUTO,
                    tooltip=(
                        'auto uses the ConvRot two-slice path when compatible and '
                        'held chunked FP8 for FP8 checkpoints. With Allow FP8 '
                        'conversion, ordinary BF16/FP16 weights may be converted '
                        'to FP8 E4M3 when supported. Preserve precision keeps '
                        'floating weights floating while retaining bounded MLP '
                        'chunking. Explicit off remains off.'
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
                # Positional compatibility tombstone. Do not remove, reorder, or
                # mark non-serializable: old workflows store this slot by index.
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
                # Must stay after every legacy serialized widget. Old workflows
                # therefore have no saved value here and adopt the new default.
                io.Combo.Input(
                    'precision_mode',
                    display_name='Precision mode',
                    options=list(PRECISION_MODE_OPTIONS),
                    default=PRECISION_MODE_PRESERVE,
                    advanced=True,
                    tooltip=(
                        'Preserve precision introduces no new weight quantization. '
                        'Allow FP8 conversion permits supported BF16/FP16 QKV and '
                        'MLP weights to be converted to FP8 E4M3 for additional '
                        'memory/performance savings.'
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
                        'Auto uses full-density Kitchen streaming when no explicit '
                        'attention selector has claimed the input model, but preserves '
                        'an explicit selector. Forced explicitly allows this node to '
                        'replace dense attention with full-density Comfy Kitchen. '
                        'An H3 Sparse Attention request is always authoritative.'
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
        # preserve_precision is intentionally ignored. It only absorbs the old
        # positional workflow value so the appended enums can migrate defaults.
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