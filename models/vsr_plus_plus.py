import torch
import torch.nn.functional as F
from torch import nn

from models.video_blocks import ResidualBlock, center_index


class _SecondOrderPropagator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cell = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

    def forward(self, features: torch.Tensor, reverse: bool = False) -> list[torch.Tensor]:
        b, t, c, h, w = features.shape
        h1 = features.new_zeros(b, c, h, w)
        h2 = features.new_zeros(b, c, h, w)
        out = [h1 for _ in range(t)]
        indices = range(t - 1, -1, -1) if reverse else range(t)
        for i in indices:
            current = self.cell(torch.cat([features[:, i], h1, h2], dim=1))
            out[i] = current
            h2, h1 = h1, current
        return out


class VSRPlusPlus(nn.Module):
    """Lightweight BasicVSR++-style model without external CUDA extensions."""

    def __init__(
        self,
        scale_factor: int = 2,
        num_channels: int = 3,
        num_frames: int = 7,
        hidden_channels: int = 64,
        num_blocks: int = 5,
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
        self.forward_prop1 = _SecondOrderPropagator(hidden_channels)
        self.backward_prop1 = _SecondOrderPropagator(hidden_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels, kernel_size=1),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(hidden_channels),
        )
        self.forward_prop2 = _SecondOrderPropagator(hidden_channels)
        self.backward_prop2 = _SecondOrderPropagator(hidden_channels)
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels, kernel_size=3, padding=1),
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
            raise ValueError(f"VSRPlusPlus expects input shape B,T,C,H,W; got {tuple(x.shape)}")
        if x.shape[1] != self.num_frames:
            raise ValueError(f"Expected {self.num_frames} frames, got {x.shape[1]}")

        features = self._encode(x)
        bwd1 = self.backward_prop1(features, reverse=True)
        fwd1 = self.forward_prop1(features, reverse=False)

        refined_frames = []
        for i in range(x.shape[1]):
            refined_frames.append(self.refine(torch.cat([features[:, i], fwd1[i], bwd1[i]], dim=1)))
        refined = torch.stack(refined_frames, dim=1)

        bwd2 = self.backward_prop2(refined, reverse=True)
        fwd2 = self.forward_prop2(refined, reverse=False)

        center = center_index(x.shape[1])
        center_feat = refined[:, center]
        attn_scores = []
        for i in range(x.shape[1]):
            attn_scores.append(self.attention(torch.cat([refined[:, i], center_feat], dim=1)))
        attn = torch.softmax(torch.stack(attn_scores, dim=1), dim=1)
        context = (attn * refined).sum(dim=1)

        fused = torch.cat([center_feat, fwd2[center], bwd2[center], context], dim=1)
        residual = F.interpolate(
            x[:, center],
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        return self.reconstruct(fused) + residual
