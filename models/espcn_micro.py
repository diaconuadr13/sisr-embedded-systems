import torch
from torch import nn


class ESPCNMicro(nn.Module):
    """Very small ESPCN variant for microcontroller-class grayscale SISR."""

    def __init__(self, scale_factor: int = 2, num_channels: int = 1) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                8,
                num_channels * (scale_factor ** 2),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.pixel_shuffle(x)
