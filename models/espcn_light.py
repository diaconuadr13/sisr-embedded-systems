import torch
from torch import nn


class ESPCNLight(nn.Module):
    """ESPCN with halved channel widths (32→16) for embedded/edge deployment."""

    def __init__(self, scale_factor: int = 2, num_channels: int = 3) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, num_channels * (scale_factor ** 2),
                       kernel_size=3, stride=1, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.pixel_shuffle(x)
        return x
