'''Probe the active Torch backend and Sparse Sage ABI in a child process.'''

import argparse
import json
from pathlib import Path
import sys


RESULT_PREFIX = 'H3_SPARSE_PROBE='
SUPPORTED_CAPABILITIES = (
    (8, 0),
    (8, 6),
    (8, 7),
    (8, 9),
    (9, 0),
    (12, 0),
)


def _backend(torch):
    if getattr(torch.version, 'hip', None):
        return 'rocm'
    if getattr(torch.version, 'cuda', None):
        return 'nvidia_cuda'
    xpu = getattr(torch, 'xpu', None)
    if xpu is not None and xpu.is_available():
        return 'xpu'
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return 'mps'
    return 'cpu'


def probe(validate_sparse):
    try:
        import torch
    except Exception as exc:
        return {
            'ok': False,
            'error': 'Torch import failed: %s: %s'
            % (type(exc).__name__, exc),
        }

    backend = _backend(torch)
    accelerator_available = bool(
        torch.cuda.is_available()
        if backend in ('nvidia_cuda', 'rocm')
        else backend in ('mps', 'xpu')
    )
    result = {
        'ok': True,
        'torch_version': str(torch.__version__),
        'cuda_version': getattr(torch.version, 'cuda', None),
        'hip_version': getattr(torch.version, 'hip', None),
        'backend': backend,
        'accelerator_available': accelerator_available,
        'capability': None,
        'device_name': backend,
        'sparse_compatible': False,
        'sparse_error': None,
    }

    if backend == 'nvidia_cuda' and accelerator_available:
        try:
            index = int(torch.cuda.current_device())
            capability = tuple(
                int(value)
                for value in torch.cuda.get_device_capability(index)
            )
            result['capability'] = list(capability)
            result['device_name'] = str(torch.cuda.get_device_name(index))
        except Exception as exc:
            result['accelerator_available'] = False
            result['sparse_error'] = 'NVIDIA CUDA probe failed: %s: %s' % (
                type(exc).__name__,
                exc,
            )
            return result

        if capability not in SUPPORTED_CAPABILITIES:
            result['sparse_error'] = (
                'Sparse Sage does not support device capability %d.%d'
                % capability
            )
            return result
        if validate_sparse:
            try:
                pack_root = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(pack_root))
                from h3_optimizations.attention.sparse.sparse_sage import (
                    load_sparse_sage_spec,
                )

                spec = load_sparse_sage_spec(
                    capability=capability,
                    cuda_version=tuple(
                        int(part)
                        for part in str(result['cuda_version']).split('.')[:2]
                    ),
                )
                result['sparse_compatible'] = True
                result['sparse_architecture'] = spec.architecture
                result['sparse_version'] = spec.version
            except Exception as exc:
                result['sparse_error'] = '%s: %s' % (
                    type(exc).__name__,
                    exc,
                )
        return result

    if backend == 'rocm':
        result['sparse_error'] = 'Sparse Sage requires NVIDIA CUDA, not ROCm'
    elif not accelerator_available and backend == 'nvidia_cuda':
        result['sparse_error'] = 'NVIDIA CUDA is not available'
    else:
        result['sparse_error'] = 'Sparse Sage requires NVIDIA CUDA; detected %s' % backend
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-sparse', action='store_true')
    args = parser.parse_args()
    print(RESULT_PREFIX + json.dumps(probe(args.validate_sparse), sort_keys=True))


if __name__ == '__main__':
    main()
