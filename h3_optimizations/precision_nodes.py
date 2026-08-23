'''Production H3 memory node with an optional preserve-precision policy.'''

from comfy_api.latest import io, ui

from .apply import apply_plan
from .plan import (
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
    MemoryRequest,
    read_plan,
)
from .status import format_memory_status

DEFAULT_CHUNK_ROWS = 4096
NODE_CATEGORY = 'H3-Optimizations/Model Patches'


def _memory_request(
    *,
    fused_qkv=FUSED_QKV_AUTO,
    mlp_memory=MLP_MEMORY_AUTO,
    chunk_rows=DEFAULT_CHUNK_ROWS,
    preserve_precision=False,
):
    '''Resolve the public node controls into one immutable memory request.'''
    if not preserve_precision:
        return MemoryRequest(
            fused_qkv=fused_qkv,
            mlp_memory=mlp_memory,
            chunk_rows=int(chunk_rows),
        )

    # Preserve-precision is a policy over the existing controls rather than a
    # second optimization stack. Dense attention and QKV stay upstream because
    # the current optimized dense QKV path feeds quantized Kitchen carriers.
    # MLP auto becomes the bounded provider that never converts floating
    # checkpoint weights to FP8. An explicit MLP off request remains off.
    return MemoryRequest(
        attention=ATTENTION_EXISTING,
        fused_qkv=FUSED_QKV_OFF,
        mlp_memory=(
            MLP_MEMORY_PRESERVE
            if mlp_memory == MLP_MEMORY_AUTO
            else mlp_memory
        ),
        chunk_rows=int(chunk_rows),
    )


class H3MemoryOptimization(io.ComfyNode):
    '''Chunked H3 memory execution with an optional no-requantization policy.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MemoryOptimization',
            display_name='H3 Memory Optimization',
            category=NODE_CATEGORY,
            description=(
                'Production memory and execution optimizations for MiniMax H3. '
                'Compatible quantized checkpoints keep their checkpoint weight '
                'precision while using specialized or chunked execution paths. '
                'For ordinary BF16/FP16 checkpoints, Auto may convert supported '
                'QKV and MLP weights to FP8 E4M3 when accelerated FP8 is '
                'available. Enable Preserve precision to forbid new quantization: '
                'dense attention and QKV stay upstream while bounded MLP and '
                'modulation chunking remain enabled.'
            ),
            search_aliases=[
                'H3 VRAM',
                'H3 memory',
                'H3 fused QKV',
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
                        'FP8 uses held FP8 projection. BF16/FP16 may be converted '
                        'to FP8 E4M3; this conversion is lossy and may change '
                        'generated output. Unsupported quantized formats use '
                        'standard Comfy QKV. off always uses standard H3 QKV. '
                        'Preserve precision overrides auto to the standard QKV path.'
                    ),
                ),
                io.Combo.Input(
                    'mlp_memory',
                    display_name='MLP memory optimization',
                    options=[MLP_MEMORY_AUTO, MLP_MEMORY_OFF],
                    default=MLP_MEMORY_AUTO,
                    tooltip=(
                        'auto uses the ConvRot two-slice path when compatible and '
                        'held chunked FP8 for FP8 checkpoints. Ordinary BF16/FP16 '
                        'weights may be converted to FP8 E4M3 when supported. With '
                        'Preserve precision enabled, auto instead keeps floating '
                        'weights floating while retaining bounded MLP chunking. '
                        'Explicit off remains off.'
                    ),
                ),
                io.Int.Input(
                    'chunk_rows',
                    display_name='MLP chunk rows',
                    default=DEFAULT_CHUNK_ROWS,
                    min=MIN_CHUNK_ROWS,
                    max=MAX_CHUNK_ROWS,
                    step=256,
                    advanced=True,
                    tooltip=(
                        'Maximum token rows processed by one MLP chunk. Larger '
                        'chunks may be faster but use more activation memory.'
                    ),
                ),
                io.Boolean.Input(
                    'preserve_precision',
                    display_name='Preserve precision',
                    default=False,
                    advanced=True,
                    tooltip=(
                        'Do not introduce new quantization. BF16/FP16 MLP weights '
                        'stay floating and are still processed in bounded chunks; '
                        'checkpoint-native quantization stays native where supported. '
                        'Dense attention and QKV projection remain on the upstream '
                        'Comfy path. This overrides QKV auto and MLP auto only; an '
                        'explicit MLP off request remains off.'
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
        preserve_precision=False,
    ):
        plan = read_plan(model).with_memory(
            _memory_request(
                fused_qkv=fused_qkv,
                mlp_memory=mlp_memory,
                chunk_rows=chunk_rows,
                preserve_precision=bool(preserve_precision),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_memory_status(patched)),
        )
