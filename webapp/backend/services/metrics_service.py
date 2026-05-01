from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

ARCH_DESCRIPTIONS: dict[str, str] = {
    "ESPCN": "Sub-pixel convolution (Shi et al., CVPR 2016). Feature extraction 64→32 channels + PixelShuffle.",
    "ESPCN_Light": "ESPCN with halved channels (32→16) for embedded deployment.",
    "FSRCNN": "Deconvolution-based (Dong et al., ECCV 2016). Shrinking + 4-layer mapping + expansion + deconv.",
    "SRCNN": "Bicubic upsample → 3-layer CNN refinement (Dong et al., ECCV 2014).",
    "EDSR_Tiny": "8 residual blocks (32 features) with residual scaling (0.1), no batch norm, PixelShuffle.",
    "CARN_M": "Cascading residual blocks with group convolutions (groups=4) and cascading fusion (mobile variant).",
}

_REF_INPUT_HW = (64, 64)
_thop_lock = threading.Lock()


def _get_model(arch: str, scale: int):
    from models import get_model

    import torch

    device = torch.device("cpu")
    return get_model(arch, scale=scale, device=device)


def count_params(arch: str, scale: int) -> int:
    model = _get_model(arch, scale)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops_m(arch: str, scale: int, input_hw: tuple[int, int] = _REF_INPUT_HW) -> float | None:
    """Return FLOPs in millions for a forward pass at input_hw resolution."""
    try:
        import torch
        from thop import profile
    except Exception:
        return None

    model = _get_model(arch, scale)
    dummy = torch.zeros(1, 3, input_hw[0], input_hw[1])
    with _thop_lock:
        try:
            flops, _params = profile(model, inputs=(dummy,), verbose=False)
        except Exception:
            return None
    return float(flops) / 1e6


@lru_cache(maxsize=64)
def arch_stats(arch: str, scale: int) -> dict[str, Any]:
    return {
        "params": count_params(arch, scale),
        "flops_m": compute_flops_m(arch, scale),
    }


def checkpoint_params(checkpoint_path: str) -> int | None:
    try:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            return int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
    except Exception:
        return None
    return None


def list_architectures() -> list[str]:
    from models import list_models

    return list_models()


def architecture_description(arch: str) -> str:
    return ARCH_DESCRIPTIONS.get(arch, "No description available.")
