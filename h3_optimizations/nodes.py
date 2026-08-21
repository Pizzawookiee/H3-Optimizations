'''Two composable production nodes for MiniMax H3 optimization.'''

from comfy_api.latest import ComfyExtension, io, ui

from .apply import apply_plan
from .plan import (
    DEFAULT_VIDEO_BUDGET,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MemoryRequest,
    SparseRequest,
    read_plan,
)
from .status import (
    format_memory_status,
    format_sparse_status,
)

DEFAULT_CHUNK_ROWS = 4096
NODE_CATEGORY = 'H3-Optimizations/Model Patches'


class H3MemoryOptimization(io.ComfyNode):
    '''Chunked Kitchen QKV, sparse fused QKV, and bounded MLP execution.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MemoryOptimization',
            display_name='H3 Memory Optimization',
            category=NODE_CATEGORY,
            description=(
                'Production memory and execution optimizations for MiniMax H3. '
                'ConvRot INT8 uses the specialized paths. Native FP8 uses held '
                'chunked FP8 execution, and ordinary BF16/FP16 H3 QKV/MLP '
                'weights may be converted to FP8 E4M3 when accelerated FP8 is '
                'available. NVFP4 and unsupported quantized formats preserve '
                'upstream Comfy execution.'
            ),
            search_aliases=[
                'H3 VRAM',
                'H3 memory',
                'H3 fused QKV',
                'H3 chunked MLP',
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
                        'ConvRot INT8 keeps its specialized path; FP8 uses held '
                        'FP8 projection; BF16/FP16 may be converted to FP8 E4M3. '
                        'Unsupported quantized formats use standard Comfy QKV. '
                        'off always uses standard H3 QKV.'
                    ),
                ),
                io.Combo.Input(
                    'mlp_memory',
                    display_name='MLP memory optimization',
                    options=[MLP_MEMORY_AUTO, MLP_MEMORY_OFF],
                    default=MLP_MEMORY_AUTO,
                    tooltip=(
                        'auto uses the ConvRot two-slice path when compatible, '
                        'held chunked FP8 for FP8 checkpoints, and FP8 E4M3 '
                        'execution for ordinary BF16/FP16 weights when supported. '
                        'NVFP4 and unsupported quantized formats remain upstream.'
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
    ):
        plan = read_plan(model).with_memory(
            MemoryRequest(
                fused_qkv=fused_qkv,
                mlp_memory=mlp_memory,
                chunk_rows=int(chunk_rows),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_memory_status(patched)),
        )


class H3SparseAttention(io.ComfyNode):
    '''Fixed-density Sparse Sage attention for MiniMax H3.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3SparseAttention',
            display_name='H3 Sparse Attention',
            category=NODE_CATEGORY,
            description=(
                'Fixed-density Sparse Sage attention for MiniMax H3. Sparse '
                'Attention is checkpoint-format independent: compatible ConvRot '
                'weights may use fused projection, while FP8, BF16, NVFP4, and '
                'other Comfy-supported checkpoints can use native QKV projection. '
                'Text, reference conditioning, audio, non-video queries, and mixed '
                'boundary tiles remain dense. If Sparse Sage is unavailable, '
                'supported NVIDIA GPUs use INT8 Triton sparse attention, then '
                'FP8 FlexAttention, before falling back to resolved dense attention.'
            ),
            search_aliases=[
                'H3 sparse',
                'H3 sparse attention',
                'MiniMax sparse',
                'Sparse Sage',
                'Sparge',
                'H3 acceleration',
            ],
            inputs=[
                io.Model.Input('model'),
                io.Float.Input(
                    'video_budget',
                    display_name='Video KV budget',
                    default=DEFAULT_VIDEO_BUDGET,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        'Fraction of pure target-video KV tiles retained per '
                        'head and pure-video query tile. The request rounds up '
                        'to a whole KV-tile count. Non-video context and mixed '
                        'boundary tiles stay dense. 1.0 keeps the full route '
                        'while still executing through Sparse Sage.'
                    ),
                ),
                io.Boolean.Input(
                    'denser_early_late_steps',
                    display_name='Denser Early/Late steps',
                    default=False,
                    tooltip=(
                        'Add 30 percentage points to the Video KV budget for '
                        'the first 2 and last 2 sampling steps, capped at 100%.'
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        video_budget=DEFAULT_VIDEO_BUDGET,
        denser_early_late_steps=False,
    ):
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                denser_early_late_steps=bool(denser_early_late_steps),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )


class H3OptimizationsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3MemoryOptimization, H3SparseAttention]
