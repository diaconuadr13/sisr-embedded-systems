import torch
import torch.nn.functional as F
from torch import nn

from models.video_blocks import center_frame


class VideoESPCN(nn.Module):
    """Small temporal ESPCN baseline for video super-resolution."""

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_frames: int = 3,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.num_channels = int(num_channels)
        self.num_frames = int(num_frames)

        in_channels = self.num_channels * self.num_frames
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(
                hidden_channels,
                self.num_channels * (self.scale_factor ** 2),
                kernel_size=3,
                padding=1,
            ),
        )
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"VideoESPCN expects input shape B,T,C,H,W; got {tuple(x.shape)}")
        if x.shape[1] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {x.shape[1]}")

        b, t, c, h, w = x.shape
        y = self.features(x.reshape(b, t * c, h, w))
        y = self.pixel_shuffle(y)
        residual = F.interpolate(
            center_frame(x),
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        return y + residual
