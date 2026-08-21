'''Backend-aware runtime detection for H3 optimization selection.'''

from __future__ import annotations

from dataclasses import dataclass

import torch


BACKEND_NVIDIA_CUDA = 'nvidia_cuda'
BACKEND_ROCM = 'rocm'
BACKEND_MPS = 'mps'
BACKEND_XPU = 'xpu'
BACKEND_CPU = 'cpu'
BACKEND_PRIVATEUSE = 'privateuse'


def _selected_device():
    try:
        import comfy.model_management as model_management

        return torch.device(model_management.get_torch_device())
    except Exception:
        if torch.cuda.is_available():
            return torch.device('cuda', torch.cuda.current_device())
        xpu = getattr(torch, 'xpu', None)
        if xpu is not None and xpu.is_available():
            return torch.device('xpu', xpu.current_device())
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')


@dataclass(frozen=True)
class RuntimeEnvironment:
    cuda_available: bool
    device_index: int | None
    capability: tuple[int, int] | None
    device_name: str
    backend: str = BACKEND_CPU

    @classmethod
    def detect(cls):
        try:
            device = _selected_device()
            if device.type == 'cuda':
                hip_version = getattr(torch.version, 'hip', None)
                cuda_version = getattr(torch.version, 'cuda', None)
                index = int(
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                name = str(torch.cuda.get_device_name(index))
                if hip_version:
                    return cls(False, index, None, name, BACKEND_ROCM)
                if not cuda_version or not torch.cuda.is_available():
                    return cls(
                        False,
                        index,
                        None,
                        'NVIDIA CUDA is unavailable on %s' % name,
                        BACKEND_NVIDIA_CUDA,
                    )
                capability = tuple(
                    int(value)
                    for value in torch.cuda.get_device_capability(index)
                )
                return cls(
                    True,
                    index,
                    capability,
                    name,
                    BACKEND_NVIDIA_CUDA,
                )
            if device.type == 'mps':
                return cls(False, None, None, str(device), BACKEND_MPS)
            if device.type == 'xpu':
                return cls(
                    False,
                    device.index,
                    None,
                    str(device),
                    BACKEND_XPU,
                )
            if device.type == 'cpu':
                return cls(False, None, None, str(device), BACKEND_CPU)
            return cls(
                False,
                device.index,
                None,
                str(device),
                BACKEND_PRIVATEUSE,
            )
        except Exception as exc:
            return cls(
                False,
                None,
                None,
                'device probe failed: %s: %s'
                % (type(exc).__name__, exc),
                BACKEND_CPU,
            )

    @property
    def architecture(self):
        if self.capability is None:
            return self.backend
        return 'sm%d%d' % self.capability
