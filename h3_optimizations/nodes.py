'''Composable production nodes for MiniMax H3 optimization.'''

from comfy_api.latest import ComfyExtension, io, ui

from .apply import apply_plan
from .mlp_sharing import MLPSharingProbeConfig
from .mlp_sharing.config import (
    DEFAULT_LAYER_TEXT,
    EXECUTION_SELECTORS,
    MLPSharingConfig,
    REMOVAL_OPTIONS,
    removal_option,
)
from .mlp_sharing.probe import install_probe
from .plan import (
    DEFAULT_EDGE_KV,
    DEFAULT_EDGE_STEPS,
    DEFAULT_VIDEO_BUDGET,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_REQUESTS,
    MemoryRequest,
    SparseRequest,
    parse_layer_video_budgets,
    read_plan,
)
from .status import (
    format_memory_status,
    format_sparse_status,
)

DEFAULT_CHUNK_ROWS = 4096
NODE_CATEGORY = 'H3-Optimizations/Model Patches'


def _video_budget_input():
    return io.Float.Input(
        'video_budget',
        display_name='Video KV budget',
        default=DEFAULT_VIDEO_BUDGET,
        min=0.01,
        max=1.0,
        step=0.01,
        tooltip=(
            'Fraction of pure target-video KV tiles retained per head and '
            'pure-video query tile. The request rounds up to a whole KV-tile '
            'count. Non-video context and mixed boundary tiles stay dense. '
            '1.0 keeps the full route while still executing through the '
            'selected sparse backend.'
        ),
    )


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


class H3MLPSharingProbe(io.ComfyNode):
    '''Output-exact local-token MLP sharing oracle.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MLPSharingProbe',
            display_name='H3 MLP Sharing Probe',
            category='H3-Optimizations/Experiments',
            description=(
                'Runs the exact chunked H3 MLP and measures counterfactual '
                '1T x 2Y x 2X target-video output sharing. Inference output is '
                'unchanged. Place it after H3 Memory Optimization.'
            ),
            inputs=[
                io.Model.Input('model'),
                io.Boolean.Input('enabled', default=True),
                io.String.Input(
                    'layers',
                    default=DEFAULT_LAYER_TEXT,
                    tooltip='Comma-separated H3 block indices to measure.',
                ),
                io.Boolean.Input(
                    'include_mean_input',
                    default=True,
                    tooltip=(
                        'Also evaluates each candidate pair mean through the '
                        'same exact MLP for mean-input reconstruction metrics.'
                    ),
                ),
                io.Int.Input(
                    'mean_batch_rows',
                    default=1024,
                    min=64,
                    max=4096,
                    step=64,
                    advanced=True,
                    tooltip='Maximum extra mean-input rows per diagnostic MLP call.',
                ),
                io.String.Input('run_tag', default='mlp-sharing-stage1'),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        layers=DEFAULT_LAYER_TEXT,
        include_mean_input=True,
        mean_batch_rows=1024,
        run_tag='mlp-sharing-stage1',
    ):
        if not enabled:
            return io.NodeOutput(model)
        config = MLPSharingProbeConfig(
            layers=layers,
            include_mean_input=bool(include_mean_input),
            mean_batch_rows=int(mean_batch_rows),
            run_tag=str(run_tag),
        )
        patched = model.clone()
        install_probe(patched, config)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(
                'Output-exact MLP sharing probe armed for layers %s'
                % ','.join(str(layer) for layer in config.layers)
            ),
        )


class H3MLPSharing(io.ComfyNode):
    '''Executable target-video MLP output sharing for quality experiments.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MLPSharing',
            display_name='H3 MLP Sharing',
            category='H3-Optimizations/Experiments',
            description=(
                'Evaluates fewer target-video MLP rows and broadcasts each '
                'representative output while retaining full-resolution attention, '
                'per-token gates, and residual streams. The first three sampler '
                'steps remain exact by default.'
            ),
            inputs=[
                io.Model.Input('model'),
                io.Boolean.Input('enabled', default=True),
                io.Combo.Input(
                    'removal_fraction',
                    display_name='MLP evaluations removed',
                    options=list(REMOVAL_OPTIONS),
                    default='50%',
                    tooltip=(
                        'Requested target-video row reduction inside complete local '
                        'cells. The report records the realized fraction after chunk '
                        'and modulation-boundary exclusions.'
                    ),
                ),
                io.Combo.Input(
                    'selector',
                    options=list(EXECUTION_SELECTORS),
                    default='input_cosine',
                ),
                io.Int.Input(
                    'start_after_step',
                    display_name='Protect first steps',
                    default=3,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        'Steps with indices below this value use the exact MLP. '
                        '3 protects sampler steps 0, 1, and 2.'
                    ),
                ),
                io.String.Input(
                    'layers',
                    default='all',
                    advanced=True,
                    tooltip='all or comma-separated H3 block indices.',
                ),
                io.Int.Input(
                    'selector_seed',
                    default=0,
                    min=0,
                    max=0x7FFFFFFF,
                    advanced=True,
                ),
                io.String.Input(
                    'run_tag',
                    default='mlp-sharing-quality',
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        removal_fraction='50%',
        selector='input_cosine',
        start_after_step=3,
        layers='all',
        selector_seed=0,
        run_tag='mlp-sharing-quality',
    ):
        if not enabled:
            return io.NodeOutput(model)
        config = MLPSharingConfig(
            selector=selector,
            removal_fraction=removal_fraction,
            start_after_step=int(start_after_step),
            layers=layers,
            selector_seed=int(selector_seed),
            run_tag=str(run_tag),
        )
        plan = read_plan(model).with_mlp_sharing(config)
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(
                'MLP sharing %s with %s after %d protected step(s)'
                % (
                    removal_option(config.removal_fraction),
                    config.selector,
                    config.start_after_step,
                )
            ),
        )


class H3MLPSharingProbeOutput(io.ComfyNode):
    '''Terminal sink for H3 video-and-audio latents in probe workflows.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MLPSharingProbeOutput',
            display_name='H3 MLP Sharing Probe Output',
            category='H3-Optimizations/Experiments',
            description=(
                'Completes an MLP sharing probe without decoding or trying to '
                'serialize the MiniMax H3 video-and-audio latent.'
            ),
            inputs=[io.Latent.Input('samples')],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, samples):
        del samples
        return io.NodeOutput(
            ui=ui.PreviewText('H3 MLP sharing probe sampling completed')
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
                _video_budget_input(),
                io.Boolean.Input(
                    'denser_early_late_steps',
                    display_name='Denser Early/Late steps',
                    default=False,
                    tooltip=(
                        'Add 30 percentage points to the Video KV budget for '
                        'the first 2 and last 2 sampling steps, capped at 100%.'
                    ),
                ),
                io.String.Input(
                    'layer_video_budgets',
                    display_name='Per-layer video KV budgets',
                    default='',
                    advanced=True,
                    tooltip=(
                        'Optional comma-separated budget fractions for all 50 H3 '
                        'layers. Applies at every sampling step and cannot be '
                        'combined with Denser Early/Late steps.'
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
        layer_video_budgets='',
    ):
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                denser_early_late_steps=bool(denser_early_late_steps),
                layer_video_budgets=parse_layer_video_budgets(
                    layer_video_budgets
                ),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )


class H3SparseAttentionAdvanced(io.ComfyNode):
    '''Sparse attention with explicit backend and edge KV budgets.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3SparseAttentionAdvanced',
            display_name='H3 Sparse Attention (Advanced)',
            category=NODE_CATEGORY,
            description=(
                'Advanced fixed-density sparse attention for MiniMax H3. '
                'Video KV budget controls middle sampling steps; Early KV and '
                'Late KV override the first and last configured step counts. '
                'Backend auto uses the normal fallback chain; explicit backend '
                'selections are hard requirements.'
            ),
            search_aliases=[
                'H3 sparse advanced',
                'H3 sparse schedule',
                'Sparse Sage advanced',
                'H3 early late KV',
                'H3 sparse backend',
            ],
            inputs=[
                io.Model.Input('model'),
                _video_budget_input(),
                io.Int.Input(
                    'early_steps',
                    display_name='Early steps',
                    default=DEFAULT_EDGE_STEPS,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip='Number of first sampling steps that use Early KV.',
                ),
                io.Float.Input(
                    'early_kv',
                    display_name='Early KV',
                    default=DEFAULT_EDGE_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip='Video KV budget used during the early-step window.',
                ),
                io.Int.Input(
                    'late_steps',
                    display_name='Late steps',
                    default=DEFAULT_EDGE_STEPS,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip='Number of final sampling steps that use Late KV.',
                ),
                io.Float.Input(
                    'late_kv',
                    display_name='Late KV',
                    default=DEFAULT_EDGE_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip='Video KV budget used during the late-step window.',
                ),
                io.Combo.Input(
                    'backend',
                    display_name='Sparse backend',
                    options=list(SPARSE_BACKEND_REQUESTS),
                    default=SPARSE_BACKEND_AUTO,
                    tooltip=(
                        'auto uses Sparse Sage, then INT8 Triton, then FP8 '
                        'FlexAttention, then the resolved dense fallback. '
                        'Explicit backend choices fail if that backend is '
                        'unavailable and do not switch to another backend. '
                        'Bypass this node to force dense attention.'
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
        early_steps=DEFAULT_EDGE_STEPS,
        early_kv=DEFAULT_EDGE_KV,
        late_steps=DEFAULT_EDGE_STEPS,
        late_kv=DEFAULT_EDGE_KV,
        backend=SPARSE_BACKEND_AUTO,
    ):
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                early_steps=int(early_steps),
                early_kv=float(early_kv),
                late_steps=int(late_steps),
                late_kv=float(late_kv),
                backend=backend,
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )


class H3OptimizationsExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            H3MemoryOptimization,
            H3MLPSharing,
            H3MLPSharingProbe,
            H3MLPSharingProbeOutput,
            H3SparseAttention,
            H3SparseAttentionAdvanced,
        ]
