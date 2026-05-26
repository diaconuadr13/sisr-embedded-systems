import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def center_index(num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    return num_frames // 2


def center_frame(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Video VSR models expect input shape B,T,C,H,W; got {tuple(x.shape)}")
    return x[:, center_index(x.shape[1])]
