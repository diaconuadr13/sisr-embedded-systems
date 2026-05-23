import torch
from torch import nn


class VideoESPCN(nn.Module):
    """Lightweight multi-frame ESPCN model.

    Frames are concatenated on the channel axis before entering the network.
    For RGB with three frames, the input has 9 channels and the output has
    3 channels after pixel shuffle.
    """

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_frames: int = 3,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        if scale_factor <= 0:
            raise ValueError("scale_factor must be > 0")
        if num_channels <= 0:
            raise ValueError("num_channels must be > 0")
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be > 0")

        self.scale_factor = int(scale_factor)
        self.num_channels = int(num_channels)
        self.num_frames = int(num_frames)
        self.hidden_channels = int(hidden_channels)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_frames * num_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.feature_extractor(x))


class VideoESPCNMicro(VideoESPCN):
    """Smaller grayscale-first video ESPCN variant."""

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 1,
        num_frames: int = 3,
        hidden_channels: int = 16,
    ) -> None:
        super().__init__(
            scale_factor=scale_factor,
            num_channels=num_channels,
            num_frames=num_frames,
            hidden_channels=hidden_channels,
        )
