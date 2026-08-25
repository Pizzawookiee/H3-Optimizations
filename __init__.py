'''ComfyUI entry point for H3 Optimizations.'''

from importlib import util as importlib_util
from pathlib import Path
import sys


def _load_h3_package():
    '''Load the inner package under one canonical Python module name.

    ComfyUI may import this node pack under a generated package name derived
    from the custom-node directory. A relative fallback import would then give
    the inner package that generated name too. Aliasing only the package object
    as ``h3_optimizations`` is insufficient because its ``__spec__`` still
    points at the generated name, so later relative imports can load sibling
    modules twice under different names. Class identity checks then fail even
    though both classes came from the same source file.

    Load the package as ``h3_optimizations`` from the start instead. Every
    relative import inside the package then resolves through the same namespace.
    '''
    try:
        import h3_optimizations
    except ModuleNotFoundError as exc:
        if exc.name != 'h3_optimizations':
            raise

        package_dir = Path(__file__).resolve().parent / 'h3_optimizations'
        spec = importlib_util.spec_from_file_location(
            'h3_optimizations',
            package_dir / '__init__.py',
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError('could not create h3_optimizations package spec')

        module = importlib_util.module_from_spec(spec)
        sys.modules['h3_optimizations'] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop('h3_optimizations', None)
            raise
        return module
    return h3_optimizations


_h3_optimizations = _load_h3_package()

from h3_optimizations.public_nodes import H3OptimizationsExtension


async def comfy_entrypoint() -> H3OptimizationsExtension:
    return H3OptimizationsExtension()
