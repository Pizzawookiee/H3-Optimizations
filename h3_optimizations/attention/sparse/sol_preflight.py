'''Health-gated preflight wrapper for native Sol residual attention.'''

from .sol_residual import SolResidualError
from .sol_residual import preflight_sol_residual as _preflight_sol_residual


def preflight_sol_residual(**kwargs):
    spec = _preflight_sol_residual(**kwargs)
    from ...native import selftest

    if not selftest.sparse_lse_check():
        raise SolResidualError(
            'Sol residual attention requires a native sparse-LSE path that '
            'passed its device self-test'
        )
    return spec
