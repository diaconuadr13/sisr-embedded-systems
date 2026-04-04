"""Model registry and factory for SISR architectures."""
from typing import Dict, Type

import torch
from torch import nn

from models.espcn import ESPCN
from models.espcn_light import ESPCNLight
from models.fsrcnn import FSRCNN

# Registry: canonical name → class. Lookup is case-insensitive.
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ESPCN": ESPCN,
    "ESPCN_Light": ESPCNLight,
    "FSRCNN": FSRCNN,
}


def get_model(model_name: str, scale: int, device: torch.device) -> nn.Module:
    """Instantiate a model by name. Raises ValueError on unknown name."""
    # Case-insensitive lookup
    lookup = {k.lower(): v for k, v in MODEL_REGISTRY.items()}
    cls = lookup.get(model_name.lower())
    if cls is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return cls(scale_factor=scale, num_channels=3).to(device)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())
