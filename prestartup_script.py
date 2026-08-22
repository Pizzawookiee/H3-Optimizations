'''Install the optional Sparse Sage backend before node registration.'''

import logging
from pathlib import Path
import runpy


LOG_PREFIX = '[H3 Optimizations]'
installer = runpy.run_path(
    str(Path(__file__).resolve().parent / 'h3_optimizations' / 'sparse_install.py')
)
ensure_sparse_sage = installer['ensure_sparse_sage']


def run_installer(*, cli=False):
    if cli:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    ready = bool(ensure_sparse_sage())
    if cli:
        if ready:
            print('%s Sparse Sage is ready' % LOG_PREFIX)
        else:
            print('%s Sparse Sage is unavailable; see messages above' % LOG_PREFIX)
    return ready


if __name__ == '__main__':
    raise SystemExit(0 if run_installer(cli=True) else 1)

run_installer()
