'''Configuration for H3 MLP output-sharing experiments.'''

from dataclasses import dataclass
import re

DEFAULT_LAYERS = (0, 1, 2, 5, 10, 20, 30, 40, 47, 49)
DEFAULT_LAYER_TEXT = ','.join(str(value) for value in DEFAULT_LAYERS)
TARGET_FRACTIONS = tuple(value / 100.0 for value in range(5, 51, 5))
SELECTORS = ('output_oracle', 'input_cosine', 'input_l2', 'random_local')
RECONSTRUCTIONS = ('representative', 'mean_input')
EXECUTION_SELECTORS = ('input_cosine', 'random_local')
REMOVAL_OPTIONS = ('0%', '25%', '50%', '75%', '87.5%')
REMOVAL_FRACTIONS = {
    '0%': 0.0,
    '25%': 0.25,
    '50%': 0.5,
    '75%': 0.75,
    '87.5%': 0.875,
}


def parse_layers(value):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(',') if part.strip()]
        if not parts:
            raise ValueError('layers must contain at least one block index')
        layers = tuple(int(part) for part in parts)
    else:
        layers = tuple(int(part) for part in value)
    if len(set(layers)) != len(layers):
        raise ValueError('layers may not contain duplicates')
    if any(layer < 0 or layer >= 50 for layer in layers):
        raise ValueError('layers must be in [0, 49]')
    return tuple(sorted(layers))


def parse_execution_layers(value):
    if isinstance(value, str) and value.strip().lower() == 'all':
        return tuple(range(50))
    return parse_layers(value)


def parse_removal_fraction(value):
    if isinstance(value, str):
        text = value.strip()
        if text in REMOVAL_FRACTIONS:
            return REMOVAL_FRACTIONS[text]
        if text.endswith('%'):
            value = float(text[:-1]) / 100.0
        else:
            value = float(text)
    fraction = float(value)
    for allowed in REMOVAL_FRACTIONS.values():
        if abs(fraction - allowed) < 1.0e-9:
            return allowed
    raise ValueError(
        'removal_fraction must be one of %s'
        % ', '.join(REMOVAL_OPTIONS)
    )


def removal_option(value):
    fraction = parse_removal_fraction(value)
    return next(
        name for name, allowed in REMOVAL_FRACTIONS.items()
        if fraction == allowed
    )


@dataclass(frozen=True)
class MLPSharingProbeConfig:
    layers: tuple = DEFAULT_LAYERS
    include_mean_input: bool = True
    mean_batch_rows: int = 1024
    run_tag: str = 'mlp-sharing-stage1'

    def __post_init__(self):
        object.__setattr__(self, 'layers', parse_layers(self.layers))
        if int(self.mean_batch_rows) < 64 or int(self.mean_batch_rows) > 4096:
            raise ValueError('mean_batch_rows must be in [64, 4096]')
        tag = str(self.run_tag)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', tag):
            raise ValueError(
                'run_tag must start with a letter or number and contain only '
                'letters, numbers, dot, underscore, or dash'
            )

    @property
    def signature(self):
        return (
            tuple(self.layers),
            bool(self.include_mean_input),
            int(self.mean_batch_rows),
            str(self.run_tag),
        )


@dataclass(frozen=True)
class MLPSharingConfig:
    selector: str = 'input_cosine'
    removal_fraction: float = 0.5
    start_after_step: int = 3
    layers: tuple = tuple(range(50))
    selector_seed: int = 0
    run_tag: str = 'mlp-sharing-quality'

    def __post_init__(self):
        if self.selector not in EXECUTION_SELECTORS:
            raise ValueError('unknown MLP sharing selector %r' % self.selector)
        object.__setattr__(
            self,
            'removal_fraction',
            parse_removal_fraction(self.removal_fraction),
        )
        object.__setattr__(self, 'layers', parse_execution_layers(self.layers))
        if isinstance(self.start_after_step, bool) or int(self.start_after_step) < 0:
            raise ValueError('start_after_step must be a non-negative integer')
        object.__setattr__(self, 'start_after_step', int(self.start_after_step))
        if isinstance(self.selector_seed, bool):
            raise ValueError('selector_seed must be an integer')
        object.__setattr__(self, 'selector_seed', int(self.selector_seed))
        tag = str(self.run_tag)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', tag):
            raise ValueError(
                'run_tag must start with a letter or number and contain only '
                'letters, numbers, dot, underscore, or dash'
            )

    @property
    def geometry(self):
        return (1, 2, 4) if self.removal_fraction == 0.875 else (1, 2, 2)

    @property
    def signature(self):
        return (
            self.selector,
            float(self.removal_fraction),
            int(self.start_after_step),
            tuple(self.layers),
            int(self.selector_seed),
            str(self.run_tag),
        )


STAGE0_KV_TILE = 64
STAGE0_SAMPLE_ROWS = 128


@dataclass(frozen=True)
class Stage0Config:
    """One dense-MLP diagnostic run over attention selection and cache reuse."""

    layers: tuple = DEFAULT_LAYERS
    measure_cache: bool = True
    sample_blocks: int = 4
    cache_step_stride: int = 1
    start_step: int = 1
    selector_seed: int = 0
    run_tag: str = 'mlp-stage0'

    def __post_init__(self):
        object.__setattr__(self, 'layers', parse_layers(self.layers))
        if isinstance(self.sample_blocks, bool):
            raise ValueError('sample_blocks must be an integer')
        if int(self.sample_blocks) < 1 or int(self.sample_blocks) > 64:
            raise ValueError('sample_blocks must be in [1, 64]')
        object.__setattr__(self, 'sample_blocks', int(self.sample_blocks))
        if isinstance(self.cache_step_stride, bool):
            raise ValueError('cache_step_stride must be an integer')
        if int(self.cache_step_stride) < 1:
            raise ValueError('cache_step_stride must be at least 1')
        object.__setattr__(self, 'cache_step_stride', int(self.cache_step_stride))
        if isinstance(self.start_step, bool) or int(self.start_step) < 0:
            raise ValueError('start_step must be a non-negative integer')
        object.__setattr__(self, 'start_step', int(self.start_step))
        if isinstance(self.selector_seed, bool):
            raise ValueError('selector_seed must be an integer')
        object.__setattr__(self, 'selector_seed', int(self.selector_seed))
        tag = str(self.run_tag)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', tag):
            raise ValueError(
                'run_tag must start with a letter or number and contain only '
                'letters, numbers, dot, underscore, or dash'
            )

    @property
    def sample_rows(self):
        return int(self.sample_blocks) * STAGE0_SAMPLE_ROWS

    @property
    def signature(self):
        return (
            tuple(self.layers),
            bool(self.measure_cache),
            int(self.sample_blocks),
            int(self.cache_step_stride),
            int(self.start_step),
            int(self.selector_seed),
            str(self.run_tag),
        )
