"""Small, randomly initialized geometry network; no lane or arrow type prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

GEOMETRY_CLASSES = ("lane", "arrow", "crosswalk", "stopline")


def _normalization(channels: int) -> nn.GroupNorm:
    # GroupNorm also works with the small batches needed on an 8 GB GPU.
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            _normalization(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _normalization(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.layers(x) + self.skip(x))


class GeometryUNet(nn.Module):
    """Residual U-Net with full-resolution independent geometry logits.

    The channel order is always ``lane, arrow, crosswalk, stopline``. Lane and
    stopline targets represent their continuous centerline ribbons. Arrow targets
    represent oriented enclosing rectangles; crosswalk targets are whole regions.
    This baseline does not learn an instance embedding or lane-type transition head.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("The intensity-only baseline requires in_channels=1")
        if base_channels < 4:
            raise ValueError("base_channels must be at least 4")
        self.enc0 = ResidualBlock(in_channels, base_channels)
        self.enc1 = ResidualBlock(base_channels, base_channels * 2)
        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.bridge = ResidualBlock(base_channels * 4, base_channels * 8)
        self.dec2 = ResidualBlock(base_channels * 12, base_channels * 4)
        self.dec1 = ResidualBlock(base_channels * 6, base_channels * 2)
        self.dec0 = ResidualBlock(base_channels * 3, base_channels)
        self.head = nn.Conv2d(base_channels, len(GEOMETRY_CLASSES), 1)

    @staticmethod
    def _join(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((x, skip), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 8:
            raise ValueError("Expected BCHW intensity tensor with C=1 and H,W >= 8")
        e0 = self.enc0(x)
        e1 = self.enc1(F.max_pool2d(e0, 2))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        bridge = self.bridge(F.max_pool2d(e2, 2))
        d2 = self.dec2(self._join(bridge, e2))
        d1 = self.dec1(self._join(d2, e1))
        return self.head(self.dec0(self._join(d1, e0)))
