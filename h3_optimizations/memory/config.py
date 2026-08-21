'''Validated configuration for bounded H3 MLP activation execution.'''

from dataclasses import dataclass

MODE_NATIVE = 'mlp_chunked_native'
MODE_FP8 = 'mlp_chunked_fp8'
MODE_CONVROT_2SLICE = 'mlp_chunked_convrot_2slice'
IMPLEMENTED_MODES = (MODE_NATIVE, MODE_FP8, MODE_CONVROT_2SLICE)
DEFAULT_MODE = MODE_NATIVE

MIN_CHUNK_ROWS = 256
MAX_CHUNK_ROWS = 65_536
DEFAULT_CHUNK_ROWS = 2_048
DEFAULT_ALIGNMENT = 256


@dataclass(frozen=True)
class ActivationMemoryConfig:
    mode: str = DEFAULT_MODE
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    alignment: int = DEFAULT_ALIGNMENT
    strict: bool = True

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                'MLP memory mode %r is unavailable; implemented modes: %s'
                % (self.mode, ', '.join(IMPLEMENTED_MODES))
            )
        if not MIN_CHUNK_ROWS <= int(self.chunk_rows) <= MAX_CHUNK_ROWS:
            raise ValueError(
                'chunk_rows must be between %d and %d, got %r'
                % (MIN_CHUNK_ROWS, MAX_CHUNK_ROWS, self.chunk_rows)
            )
        if int(self.alignment) <= 0:
            raise ValueError('alignment must be positive')
        if int(self.chunk_rows) < int(self.alignment):
            raise ValueError('chunk_rows must be at least alignment')
        if int(self.chunk_rows) % int(self.alignment):
            raise ValueError(
                'chunk_rows (%d) must be a multiple of alignment (%d)'
                % (self.chunk_rows, self.alignment)
            )

    @property
    def native_swiglu(self):
        return self.mode == MODE_NATIVE

    @property
    def fp8(self):
        return self.mode == MODE_FP8

    @property
    def convrot_2slice(self):
        return self.mode == MODE_CONVROT_2SLICE

    @property
    def signature(self):
        return (
            self.mode,
            int(self.chunk_rows),
            int(self.alignment),
            bool(self.strict),
        )
