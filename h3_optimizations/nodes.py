'''Composable production nodes for MiniMax H3 optimization.'''

from comfy_api.latest import io, ui

from .apply import apply_plan
from .mlp_sharing import MLPSharingProbeConfig
from .mlp_sharing.config import (
    DEFAULT_LAYER_TEXT,
    EXECUTION_SELECTORS,
    MLPSharingConfig,
    REMOVAL_OPTIONS,
    removal_option,
)
from .mlp_sharing.config import Stage0Config
from .mlp_sharing.probe import install_probe
from .mlp_sharing.stage0 import install_stage0
from .ordering_probe import (
    AttentionOrderingConfig,
    DEFAULT_BUDGETS as ORDERING_DEFAULT_BUDGETS,
    DEFAULT_LAYERS as ORDERING_DEFAULT_LAYERS,
    DEFAULT_STEPS as ORDERING_DEFAULT_STEPS,
    install_ordering_probe,
)
from .plan import (
    ATTENTION_EXISTING,
    DEFAULT_EDGE_KV,
    DEFAULT_EDGE_STEPS,
    DEFAULT_VIDEO_BUDGET,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MAX_CHUNK_ROWS,
    MIN_CHUNK_ROWS,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MLP_MEMORY_PRESERVE,
    SPARSE_BACKEND_COMPAT_REQUESTS,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_PUBLIC_REQUESTS,
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
        display_name='Video attention budget',
        default=DEFAULT_VIDEO_BUDGET,
        min=0.01,
        max=1.0,
        step=0.01,
        tooltip=(
            'Controls the speed/quality tradeoff for target-video attention. '
            'Lower values are faster but retain fewer video attention connections '
            'and can reduce prompt adherence, change motion/detail, or otherwise '
            'change the result. There is no universally safe value: some prompts '
            'tolerate very low budgets while others require substantially more. '
            'The request rounds up to whole KV tiles; non-video context and mixed '
            'boundary tiles stay dense. 1.0 retains the full video route.'
        ),
    )


def _memory_request(
    *,
    fused_qkv=FUSED_QKV_AUTO,
    mlp_memory=MLP_MEMORY_AUTO,
    chunk_rows=DEFAULT_CHUNK_ROWS,
    preserve_precision=True,
):
    '''Resolve the public memory controls into one immutable request.'''
    if not preserve_precision:
        return MemoryRequest(
            fused_qkv=fused_qkv,
            mlp_memory=mlp_memory,
            chunk_rows=int(chunk_rows),
        )

    # Preserve precision is a policy over the normal memory node rather than a
    # separate optimization stack. Dense attention stays upstream, while an
    # explicitly selected sparse backend may consume checkpoint-native bounded
    # QKV projection without introducing weight conversion. MLP auto still uses
    # bounded execution, but never converts floating weights to FP8.
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
    '''Chunked Kitchen QKV, sparse fused QKV, and bounded MLP execution.'''

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
                'available. Preserve precision defaults on to forbid new weight '
                'quantization: compatible ConvRot INT8 QKV streams BF16 chunks '
                'to sparse Kitchen, while other dense QKV stays upstream and '
                'bounded MLP and modulation chunking remain enabled.'
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
                        'With sparse Kitchen, Preserve precision streams compatible '
                        'ConvRot INT8 projection through BF16 chunks into its INT8 '
                        'carrier; other formats use their precision-preserving path.'
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
                    display_name='Activation chunk rows',
                    default=DEFAULT_CHUNK_ROWS,
                    min=MIN_CHUNK_ROWS,
                    max=MAX_CHUNK_ROWS,
                    step=256,
                    advanced=True,
                    tooltip=(
                        'Maximum token rows processed by one MLP or FinalLayer '
                        'chunk. Larger chunks may be faster but use more '
                        'activation memory.'
                    ),
                ),
                io.Boolean.Input(
                    'preserve_precision',
                    display_name='Preserve precision',
                    default=True,
                    advanced=True,
                    tooltip=(
                        'Do not introduce new quantization. BF16/FP16 MLP weights '
                        'stay floating and are still processed in bounded chunks; '
                        'checkpoint-native quantization stays native where supported. '
                        'Compatible ConvRot INT8 QKV streams BF16 projection, norm, '
                        'and RoPE chunks into a selected sparse Kitchen carrier. '
                        'Other dense QKV remains on the upstream Comfy path. This '
                        'overrides QKV auto and MLP auto only; an explicit MLP off '
                        'request remains off.'
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


class H3AttentionOrderingProbe(io.ComfyNode):
    '''Compare post-RoPE target-video traversal orders without changing token order.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3AttentionOrderingProbe',
            display_name='H3 Attention Ordering Probe',
            category='H3-Optimizations/Experiments',
            description=(
                'Compares native, explicit row-major, time-major, Morton, and '
                'enclosing-cube Hilbert target-video ordering at identical '
                'fixed 128Q x 64KV density. Q/K/V are observed after RoPE; '
                'the model keeps its production token order. Place this node '
                'after H3 Sparse Attention.'
            ),
            inputs=[
                io.Model.Input('model'),
                io.Boolean.Input('enabled', default=True),
                io.String.Input(
                    'layers',
                    default=','.join(str(value) for value in ORDERING_DEFAULT_LAYERS),
                    tooltip='Comma-separated H3 block indices to measure.',
                ),
                io.String.Input(
                    'steps',
                    default=','.join(str(value) for value in ORDERING_DEFAULT_STEPS),
                    tooltip='Comma-separated zero-based sampler steps to measure.',
                ),
                io.String.Input(
                    'video_budgets',
                    default=','.join('%.0f' % (100.0 * value) for value in ORDERING_DEFAULT_BUDGETS),
                    tooltip='Target-video KV percentages compared at identical tile density.',
                ),
                io.Int.Input(
                    'query_samples',
                    default=64,
                    min=1,
                    max=1024,
                    tooltip='Same native target-video query tokens sampled for every ordering.',
                ),
                io.Int.Input(
                    'head_chunk',
                    default=2,
                    min=1,
                    max=56,
                    advanced=True,
                    tooltip='Attention heads analyzed together. Lower values reduce probe VRAM.',
                ),
                io.Boolean.Input(
                    'capture_uncond',
                    default=False,
                    advanced=True,
                    tooltip='Also measure the negative/unconditional branch.',
                ),
                io.String.Input('run_tag', default='attention-ordering'),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        layers=','.join(str(value) for value in ORDERING_DEFAULT_LAYERS),
        steps=','.join(str(value) for value in ORDERING_DEFAULT_STEPS),
        video_budgets=','.join('%.0f' % (100.0 * value) for value in ORDERING_DEFAULT_BUDGETS),
        query_samples=64,
        head_chunk=2,
        capture_uncond=False,
        run_tag='attention-ordering',
    ):
        if not enabled:
            return io.NodeOutput(model)
        config = AttentionOrderingConfig(
            layers=layers,
            steps=steps,
            budgets=video_budgets,
            query_samples=int(query_samples),
            head_chunk=int(head_chunk),
            capture_uncond=bool(capture_uncond),
            run_tag=str(run_tag),
        )
        patched = model.clone()
        install_ordering_probe(patched, config)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(
                'Post-RoPE ordering probe armed for layers %s, steps %s'
                % (
                    ','.join(str(value) for value in config.layers),
                    ','.join(str(value) for value in config.steps),
                )
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
    '''Fixed-density sparse attention for MiniMax H3.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3SparseAttention',
            display_name='H3 Sparse Attention',
            category=NODE_CATEGORY,
            description=(
                'Fixed-density sparse attention for MiniMax H3. Lower video '
                'attention budgets are faster but can reduce prompt adherence, '
                'change motion/detail, or otherwise change the generated result; '
                'no percentage is lossless for every prompt. Text, reference '
                'conditioning, audio, non-video queries, and mixed boundary tiles '
                'remain dense. Backend auto prefers native Kitchen INT8, then '
                'Sparse Sage, INT8 Triton, FP8 FlexAttention, and finally the '
                'resolved dense attention path.'
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
                        'Adds 30 percentage points to the video attention budget '
                        'for the first 2 and last 2 sampling steps, capped at 100%. '
                        'H3 is especially sensitive to reduced attention in early '
                        'denoising, so this can preserve prompt/timeline adherence '
                        'better than using the same low budget throughout.'
                    ),
                ),
                io.String.Input(
                    'layer_video_budgets',
                    display_name='Per-layer video KV budgets',
                    default='',
                    optional=True,
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
                'Video attention budget controls middle sampling steps; Early KV '
                'and Late KV override the first and last configured step counts. '
                'Lower budgets are faster but can change the generated result, and '
                'the quality cost depends on the prompt and where attention is '
                'removed in the denoising schedule. Kitchen INT8 64x64 is the '
                'default; FROST BF16, Sparse Sage, INT8 Triton, and FP8 '
                'FlexAttention are available as explicit alternatives.'
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
                    tooltip=(
                        'Number of first sampling steps that use Early KV. H3 is '
                        'especially sensitive to reduced attention early in denoising.'
                    ),
                ),
                io.Float.Input(
                    'early_kv',
                    display_name='Early KV',
                    default=DEFAULT_EDGE_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        'Video attention budget used during the early-step window. '
                        'Increasing this can preserve prompt/timeline adherence at '
                        'the cost of speed; lowering it is especially risky for H3.'
                    ),
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
                    tooltip=(
                        'Video attention budget used during the late-step window. '
                        'Higher values retain more exact video attention at the '
                        'cost of speed.'
                    ),
                ),
                io.Combo.Input(
                    'backend',
                    display_name='Sparse backend',
                    options=list(SPARSE_BACKEND_PUBLIC_REQUESTS),
                    default=SPARSE_BACKEND_KITCHEN,
                    tooltip=(
                        'Kitchen INT8 uses the shipped native 64Q x 64KV path. '
                        'FROST BF16 uses 64Q x 64KV routing and is available '
                        'only on SM89. '
                        'INT8 Triton and FP8 FlexAttention use the same 64Q x '
                        '64KV routing geometry. Sparse Sage uses its installed '
                        'kernel geometry. Each alternative is selected explicitly. '
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
        backend=SPARSE_BACKEND_KITCHEN,
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

    @classmethod
    def validate_inputs(cls, backend):
        if backend in SPARSE_BACKEND_COMPAT_REQUESTS:
            return True
        return 'unknown sparse backend %r' % backend


class H3MLPStage0(io.ComfyNode):
    '''Dense-MLP diagnostic for attention selection and cache reuse.'''

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='H3MLPStage0',
            display_name='H3 MLP Stage 0 Probe',
            category='H3-Optimizations/Experiments',
            description=(
                'Runs the exact chunked H3 MLP and measures whether attention '
                'signals identify disposable MLP work, how concentrated the '
                'cross-step SwiGLU delta is, and whether an FP8 cache survives '
                'it. Inference output is unchanged. Place it after H3 Memory '
                'Optimization and keep H3 Sparse Attention enabled so the '
                'attention route exists.'
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
                    'measure_cache',
                    default=True,
                    tooltip=(
                        'Also measure cross-step SwiGLU delta structure, the '
                        'AdaLN share of it, and FP8 cache error on a small '
                        'fixed sample of target-video blocks.'
                    ),
                ),
                io.Int.Input(
                    'sample_blocks',
                    default=4,
                    min=1,
                    max=64,
                    tooltip=(
                        'Number of 128-row target-video blocks held across '
                        'steps for the cache measurements.'
                    ),
                ),
                io.Int.Input(
                    'start_step',
                    default=1,
                    min=0,
                    max=1000,
                    advanced=True,
                    tooltip='First sampler step that carries cache state.',
                ),
                io.Int.Input(
                    'cache_step_stride',
                    default=1,
                    min=1,
                    max=64,
                    advanced=True,
                    tooltip='Refresh the held sample every N sampler steps.',
                ),
                io.Int.Input(
                    'selector_seed',
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    advanced=True,
                ),
                io.String.Input('run_tag', default='mlp-stage0'),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        layers=DEFAULT_LAYER_TEXT,
        measure_cache=True,
        sample_blocks=4,
        start_step=1,
        cache_step_stride=1,
        selector_seed=0,
        run_tag='mlp-stage0',
    ):
        if not enabled:
            return io.NodeOutput(model)
        config = Stage0Config(
            layers=layers,
            measure_cache=bool(measure_cache),
            sample_blocks=int(sample_blocks),
            start_step=int(start_step),
            cache_step_stride=int(cache_step_stride),
            selector_seed=int(selector_seed),
            run_tag=str(run_tag),
        )
        patched = model.clone()
        install_stage0(patched, config)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(
                'Stage 0 armed for layers %s; cache measurement %s'
                % (
                    ','.join(str(layer) for layer in config.layers),
                    'on (%d rows)' % config.sample_rows
                    if config.measure_cache
                    else 'off',
                )
            ),
        )
