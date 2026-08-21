'''Validated fixed-density Sparse Sage configuration.'''

from dataclasses import dataclass
import math

MODE_SAGE128 = 'sage128'
MODE_SAGE128_FUSED_QKV = 'sage128_fused_qkv'
IMPLEMENTED_MODES = (MODE_SAGE128, MODE_SAGE128_FUSED_QKV)
DENSITY_FIXED = 'fixed'

DENSER_EARLY_LATE_STEP_COUNT = 2
DENSER_EARLY_LATE_BONUS = 0.30


@dataclass(frozen=True)
class HybridSparseConfig:
    mode: str = MODE_SAGE128
    video_budget: float = 0.5
    strict: bool = True
    density_mode: str = DENSITY_FIXED
    denser_early_late_steps: bool = False

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                'sparse mode %r is unavailable; implemented modes: %s'
                % (self.mode, ', '.join(IMPLEMENTED_MODES))
            )
        budget = float(self.video_budget)
        if not math.isfinite(budget) or not 0.01 <= budget <= 1.0:
            raise ValueError('video_budget must be finite and in [0.01, 1]')
        if self.density_mode != DENSITY_FIXED:
            raise ValueError('only fixed Sparse Sage density is supported')

    @property
    def signature(self):
        return (
            self.mode,
            float(self.video_budget),
            bool(self.strict),
            self.density_mode,
            bool(self.denser_early_late_steps),
        )


def resolve_video_budget(config, step_index, total_steps):
    budget = float(config.video_budget)
    if not config.denser_early_late_steps:
        return budget
    step_index = int(step_index)
    total_steps = int(total_steps)
    if step_index < 0 or total_steps <= 0 or step_index >= total_steps:
        return budget
    if (
        step_index < DENSER_EARLY_LATE_STEP_COUNT
        or step_index >= total_steps - DENSER_EARLY_LATE_STEP_COUNT
    ):
        return min(1.0, budget + DENSER_EARLY_LATE_BONUS)
    return budget
