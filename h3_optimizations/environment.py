"""Minimal CUDA environment detection for H3 Sage optimization selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeEnvironment:
    cuda_available: bool
    device_index: int | None
    capability: tuple[int, int] | None
    device_name: str

    @classmethod
    def detect(cls):
        if not torch.cuda.is_available():
            return cls(False, None, None, "no CUDA device")
        try:
            try:
                import comfy.model_management as model_management

                device = torch.device(model_management.get_torch_device())
            except Exception:
                device = torch.device("cuda", torch.cuda.current_device())
            if device.type != "cuda":
                return cls(
                    False, None, None, "ComfyUI model device is %s" % device
                )
            index = int(
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            capability = tuple(
                int(value)
                for value in torch.cuda.get_device_capability(index)
            )
            return cls(
                True,
                index,
                capability,
                str(torch.cuda.get_device_name(index)),
            )
        except Exception as exc:
            return cls(
                False,
                None,
                None,
                "CUDA probe failed: %s: %s"
                % (type(exc).__name__, exc),
            )

    @property
    def architecture(self):
        if self.capability is None:
            return "none"
        return "sm%d%d" % self.capability
