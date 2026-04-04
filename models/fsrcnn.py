import torch
from torch import nn


class FSRCNN(nn.Module):
    """Dong et al. "Accelerating the Super-Resolution CNN" (ECCV 2016).
    d=56, s=12, m=4 (paper defaults)."""

    def __init__(self, scale_factor: int = 2, num_channels: int = 3,
                 d: int = 56, s: int = 12, m: int = 4) -> None:
        super().__init__()
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d),
        )
        # Shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )
        # Non-linear mapping: m Conv2d 3x3 layers
        mapping_layers: list[nn.Module] = []
        for _ in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s),
            ])
        self.mapping = nn.Sequential(*mapping_layers)
        # Expanding
        self.expanding = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        )
        # Deconvolution (learnable upsampling)
        self.deconv = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9, stride=scale_factor,
            padding=4, output_padding=scale_factor - 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x
