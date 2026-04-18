import torch
from torch import nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x) * self.res_scale


class EDSRTiny(nn.Module):
    """Stripped-down EDSR (Lim et al. CVPR 2017 workshop).

    Key differences from original EDSR:
    - No batch normalisation (removed as it hurts SR quality)
    - Residual scaling (0.1) for training stability
    - Sub-pixel convolution (pixel shuffle) for upsampling
    - Reduced width (32 features) and depth (8 blocks) for embedded deployment

    Architecture: head → N × ResBlock → conv → global residual → upsample
    """

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_feats: int = 32,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(num_feats) for _ in range(num_blocks)])
        self.body_end = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        # Global residual connection over the residual body
        feat = feat + self.body_end(self.body(feat))
        return self.upsample(feat)
