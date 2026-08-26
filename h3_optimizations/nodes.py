'''Composable production nodes for MiniMax H3 optimization.'''

from comfy_api.latest import io, ui

from .apply import apply_plan
from .node_constants import NODE_CATEGORY
from .plan import (
    DEFAULT_EDGE_KV,
    DEFAULT_EDGE_STEPS,
    DEFAULT_VIDEO_BUDGET,
    SPARSE_BACKEND_COMPAT_REQUESTS,
    SPARSE_BACKEND_KITCHEN,
    SPARSE_BACKEND_PUBLIC_REQUESTS,
    SparseRequest,
    parse_layer_video_budgets,
    read_plan,
)
from .status import (
    format_memory_status,
    format_sparse_status,
)

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
                'Sparse Sage, BF16 Triton, FP8 FlexAttention, and finally the '
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
                'default; FROST BF16, Sparse Sage, BF16 Triton, and FP8 '
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
                        'BF16 Triton and FP8 FlexAttention use the same 64Q x '
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
