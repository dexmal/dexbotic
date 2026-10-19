"""Device selection helpers shared by the inference paths.

Detection order is XPU, then CUDA, then CPU. An explicit device always wins and
is never swapped for another backend.
"""

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    for device_type in ("xpu", "cuda"):
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the first device of the best backend, else honour ``device``."""
    if device is None:
        device_type = default_device_type()
        return torch.device(device_type, 0) if device_type != "cpu" else torch.device("cpu")
    device = torch.device(device)
    if device.type in ("xpu", "cuda") and device.index is None:
        device = torch.device(device.type, 0)
    return device


def device_type(device: Union[DeviceLike, torch.Tensor]) -> str:
    """Backend type (``xpu``, ``cuda`` or ``cpu``) of a device, string or tensor."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def autocast_device_type(device: Union[DeviceLike, torch.Tensor]) -> str:
    """Device type for ``torch.autocast``; CPU for anything that is not xpu or cuda."""
    resolved = device_type(device)
    return resolved if resolved in ("xpu", "cuda") else "cpu"


def manual_seed_all(seed: int) -> None:
    """Seed the RNG of every visible accelerator that is available."""
    if is_available("xpu"):
        torch.xpu.manual_seed_all(seed)
    if is_available("cuda"):
        torch.cuda.manual_seed_all(seed)


def empty_cache(device: Union[DeviceLike, torch.Tensor] = None) -> None:
    """Release cached allocator blocks of ``device``'s backend; no-op on CPU."""
    resolved: Optional[str] = device_type(device) if device is not None else default_device_type()
    if resolved == "xpu" and is_available("xpu"):
        torch.xpu.empty_cache()
    elif resolved == "cuda" and is_available("cuda"):
        torch.cuda.empty_cache()
