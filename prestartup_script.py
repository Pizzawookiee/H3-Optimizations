'''Install the optional Sparse Sage backend before node registration.'''

from pathlib import Path
import runpy


installer = runpy.run_path(
    str(Path(__file__).resolve().parent / 'h3_optimizations' / 'sparse_install.py')
)
installer['ensure_sparse_sage']()
