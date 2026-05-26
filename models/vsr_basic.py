import torch
import torch.nn.functional as F
from torch import nn

from models.video_blocks import ResidualBlock, center_index


class VSRBasic(nn.Module):
    """Dependency-free BasicVSR-style recurrent model."""

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_frames: int = 5,
        hidden_channels: int = 48,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.num_channels = int(num_channels)
        self.num_frames = int(num_frames)
        self.hidden_channels = int(hidden_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)],
        )
        self.forward_cell = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(hidden_channels),
        )
        self.backward_cell = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(hidden_channels),
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            *[ResidualBlock(hidden_channels) for _ in range(max(1, num_blocks // 2))],
            nn.Conv2d(hidden_channels, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        features = self.encoder(x.reshape(b * t, c, h, w))
        return features.reshape(b, t, self.hidden_channels, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"VSRBasic expects input shape B,T,C,H,W; got {tuple(x.shape)}")
        if x.shape[1] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {x.shape[1]}")

        features = self._encode(x)
        b, t, f, h, w = features.shape
        center = center_index(t)
        zeros = features.new_zeros(b, f, h, w)

        forward_states = []
        state = zeros
        for i in range(t):
            state = self.forward_cell(torch.cat([features[:, i], state], dim=1))
            forward_states.append(state)

        backward_states = [zeros for _ in range(t)]
        state = zeros
        for i in reversed(range(t)):
            state = self.backward_cell(torch.cat([features[:, i], state], dim=1))
            backward_states[i] = state

        fused = torch.cat(
            [features[:, center], forward_states[center], backward_states[center]],
            dim=1,
        )
        residual = F.interpolate(
            x[:, center],
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        return self.reconstruct(fused) + residual
