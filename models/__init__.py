"""Model registry and factory for SISR architectures."""
from typing import Dict, Type

import torch
from torch import nn

from models.espcn import ESPCN
from models.espcn_light import ESPCNLight
from models.espcn_micro import ESPCNMicro
from models.fsrcnn import FSRCNN
from models.srcnn import SRCNN
from models.edsr_tiny import EDSRTiny
from models.carn_m import CARNM

# Registry: canonical name → class. Lookup is case-insensitive.
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ESPCN": ESPCN,
    "ESPCN_Light": ESPCNLight,
    "ESPCN_Micro": ESPCNMicro,
    "FSRCNN": FSRCNN,
    "SRCNN": SRCNN,
    "EDSR_Tiny": EDSRTiny,
    "CARN_M": CARNM,
}


def get_model(model_name: str, scale: int, device: torch.device,
              num_channels: int = 3) -> nn.Module:
    """Instantiate a model by name. Raises ValueError on unknown name."""
    lookup = {k.lower(): v for k, v in MODEL_REGISTRY.items()}
    cls = lookup.get(model_name.lower())
    if cls is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return cls(scale_factor=scale, num_channels=num_channels).to(device)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())
