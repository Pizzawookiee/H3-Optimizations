'''Validated fixed-density Sparse Sage configuration.'''

from dataclasses import dataclass
import math

MODE_SAGE128 = 'sage128'
MODE_SAGE128_FUSED_QKV = 'sage128_fused_qkv'
IMPLEMENTED_MODES = (MODE_SAGE128, MODE_SAGE128_FUSED_QKV)
DENSITY_FIXED = 'fixed'

DENSER_EARLY_LATE_STEP_COUNT = 2
DENSER_EARLY_LATE_BONUS = 0.30


def _validate_budget(name, value):
    if value is None:
        return
    budget = float(value)
    if not math.isfinite(budget) or not 0.01 <= budget <= 1.0:
        raise ValueError('%s must be finite and in [0.01, 1]' % name)


@dataclass(frozen=True)
class HybridSparseConfig:
    mode: str = MODE_SAGE128
    video_budget: float = 0.5
    strict: bool = True
    density_mode: str = DENSITY_FIXED
    denser_early_late_steps: bool = False
    early_steps: int | None = None
    early_kv: float | None = None
    late_steps: int | None = None
    late_kv: float | None = None

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                'sparse mode %r is unavailable; implemented modes: %s'
                % (self.mode, ', '.join(IMPLEMENTED_MODES))
            )
        _validate_budget('video_budget', self.video_budget)
        _validate_budget('early_kv', self.early_kv)
        _validate_budget('late_kv', self.late_kv)
        if self.density_mode != DENSITY_FIXED:
            raise ValueError('only fixed Sparse Sage density is supported')
        values = (self.early_steps, self.early_kv, self.late_steps, self.late_kv)
        if any(value is not None for value in values):
            if not all(value is not None for value in values):
                raise ValueError('explicit early/late sparse schedule is incomplete')
            if self.denser_early_late_steps:
                raise ValueError('legacy and explicit early/late schedules cannot be combined')
            for name, value in (
                ('early_steps', self.early_steps),
                ('late_steps', self.late_steps),
            ):
                if isinstance(value, bool) or int(value) != value or int(value) < 0:
                    raise ValueError('%s must be a non-negative integer' % name)

    @property
    def signature(self):
        return (
            self.mode,
            float(self.video_budget),
            bool(self.strict),
            self.density_mode,
            bool(self.denser_early_late_steps),
            None if self.early_steps is None else int(self.early_steps),
            None if self.early_kv is None else float(self.early_kv),
            None if self.late_steps is None else int(self.late_steps),
            None if self.late_kv is None else float(self.late_kv),
        )


def resolve_video_budget(config, step_index, total_steps):
    budget = float(config.video_budget)
    step_index = int(step_index)
    total_steps = int(total_steps)
    if step_index < 0 or total_steps <= 0 or step_index >= total_steps:
        return budget

    if config.early_steps is not None:
        in_early = step_index < int(config.early_steps)
        in_late = step_index >= total_steps - int(config.late_steps)
        if in_early and in_late:
            return max(float(config.early_kv), float(config.late_kv))
        if in_early:
            return float(config.early_kv)
        if in_late:
            return float(config.late_kv)
        return budget

    if not config.denser_early_late_steps:
        return budget
    if (
        step_index < DENSER_EARLY_LATE_STEP_COUNT
        or step_index >= total_steps - DENSER_EARLY_LATE_STEP_COUNT
    ):
        return min(1.0, budget + DENSER_EARLY_LATE_BONUS)
    return budget
