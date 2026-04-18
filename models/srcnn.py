import torch
import torch.nn.functional as F
from torch import nn


class SRCNN(nn.Module):
    """Dong et al. "Learning a Deep Convolutional Network for Image Super-Resolution" (ECCV 2014 / TPAMI 2015).

    Upsamples the LR input with bicubic interpolation, then applies 3-conv refinement
    in the HR space (patch extraction → non-linear mapping → reconstruction).
    """

    def __init__(self, scale_factor: int = 2, num_channels: int = 3) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.layers = nn.Sequential(
            # Patch extraction & representation
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            # Non-linear mapping
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Reconstruction
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        return self.layers(x)
