'''Production H3 memory node that never introduces new quantization.'''

from comfy_api.latest import io, ui

from .apply import apply_plan
from .plan import (
    ATTENTION_EXISTING,
    FUSED_QKV_OFF,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_PRESERVE,
    MemoryRequest,
    read_plan,
)
from .status import format_memory_status

DEFAULT_CHUNK_ROWS = 4096
NODE_CATEGORY = 'H3-Optimizations/Model Patches'


class H3MemoryOptimizationPreservePrecision(io.ComfyNode):
    '''Bound H3 activation memory without changing checkpoint precision.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MemoryOptimizationPreservePrecision',
            display_name='H3 Memory Optimization (Preserve Precision)',
            category=NODE_CATEGORY,
            description=(
                'Memory optimizations for MiniMax H3 that do not introduce a '
                'new quantization step. Existing dense attention and standard '
                'H3 QKV projection are preserved. The MLP and per-token '
                'modulation path use bounded token chunks while keeping the '
                'checkpoint weight format: BF16/FP16 stays floating, and '
                'checkpoint-native INT8/FP8/W4A8 remains native when a compatible '
                'bounded provider exists. Unsupported formats preserve upstream '
                'Comfy execution.'
            ),
            search_aliases=[
                'H3 lossless memory',
                'H3 preserve precision',
                'H3 BF16 memory',
                'H3 non quantized memory',
                'H3 no requant memory',
            ],
            inputs=[
                io.Model.Input('model'),
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
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        chunk_rows=DEFAULT_CHUNK_ROWS,
    ):
        plan = read_plan(model).with_memory(
            MemoryRequest(
                attention=ATTENTION_EXISTING,
                fused_qkv=FUSED_QKV_OFF,
                mlp_memory=MLP_MEMORY_PRESERVE,
                chunk_rows=int(chunk_rows),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_memory_status(patched)),
        )
