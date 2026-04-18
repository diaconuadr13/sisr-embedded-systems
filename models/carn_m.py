import torch
from torch import nn


class _GroupResBlock(nn.Module):
    """Residual block using group convolutions to reduce parameter count (CARN-M style)."""

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class _CascadeUnit(nn.Module):
    """Cascading unit: applies a SHARED residual block k times, fusing all intermediate
    feature maps via 1×1 convolutions after each application.

    Weight sharing (single shared_block reused k times) reduces parameters while the
    cascading fusion (concatenation + 1×1 conv) preserves information flow.
    """

    def __init__(self, channels: int, num_blocks: int = 3, groups: int = 4) -> None:
        super().__init__()
        # Single shared block applied num_blocks times
        self.shared_block = _GroupResBlock(channels, groups)
        # After block i+1 we concatenate (i+2) feature maps and fuse back to `channels`
        self.fuse = nn.ModuleList(
            [nn.Conv2d((i + 2) * channels, channels, kernel_size=1) for i in range(num_blocks)]
        )
        self.num_blocks = num_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        history = [x]
        current = x
        for i in range(self.num_blocks):
            current = self.shared_block(current)
            history.append(current)
            current = self.fuse[i](torch.cat(history, dim=1))
            history[-1] = current  # replace last entry with fused output
        return current


class CARNM(nn.Module):
    """Ahn et al. "Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network"
    (ECCV 2018) — mobile (CARN-M) variant.

    Key design choices:
    - Group convolutions (groups=4) in all 3×3 residual convs → fewer multiply-adds
    - Weight sharing within each cascade unit → fewer parameters
    - Two-level cascading (within unit + across units) → rich feature reuse
    - Sub-pixel upsampling at the tail (efficient on embedded hardware)
    """

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_feats: int = 64,
        num_units: int = 3,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=1)
        self.units = nn.ModuleList(
            [_CascadeUnit(num_feats, num_blocks=3, groups=4) for _ in range(num_units)]
        )
        # Global cascading across units: after unit i+1 fuse (i+2) maps → num_feats
        self.unit_fuse = nn.ModuleList(
            [nn.Conv2d((i + 2) * num_feats, num_feats, kernel_size=1) for i in range(num_units)]
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        history = [feat]
        current = feat
        for i, unit in enumerate(self.units):
            current = unit(current)
            history.append(current)
            current = self.unit_fuse[i](torch.cat(history, dim=1))
            history[-1] = current
        return self.upsample(current)
