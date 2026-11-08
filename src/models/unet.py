"""UNet implemented from scratch (Ronneberger et al., 2015).

This is the pure-CNN baseline for the comparison. The design follows the original paper:
a contracting encoder that captures context, an expanding decoder that recovers spatial
resolution, and skip connections that carry high-resolution detail from encoder to decoder.

Two small modernisations are used so the implementation stays simple and 224x224-friendly:

* convolutions use padding=1 ("same" convolutions), so feature maps keep their spatial size
  within a stage and encoder/decoder tensors line up without the cropping the original paper
  needed;
* batch normalisation follows each convolution, which stabilises training.

Input : (B, 3, 224, 224) RGB image
Output: (B, 1, 224, 224) raw logits (apply a sigmoid for probabilities)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """The repeating unit of UNet: two 3x3 convolutions, each followed by BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """One encoder step: halve the resolution with max-pooling, then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """One decoder step: upsample, concatenate the matching skip, then DoubleConv.

    The skip connection is the core idea of UNet: the decoder is given the encoder's
    high-resolution features from the same level so it can place boundaries precisely,
    instead of relying only on the coarse bottleneck representation.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            # Parameter-free upsampling, then a conv halves the channel count.
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            # Learnable upsampling that also halves the channel count.
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
        # After concatenation the channel count is (in_channels // 2) from the upsampled
        # path plus (in_channels // 2) from the skip = in_channels.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.reduce(self.up(x))
        # Guard against off-by-one sizes (if the input is not divisible by 16); pad the
        # upsampled map to the skip's spatial size before concatenating.
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)  # concatenate along channels
        return self.conv(x)


class UNet(nn.Module):
    """UNet for binary skin-lesion segmentation.

    Encoder channel widths double at each level (64 -> 128 -> 256 -> 512) and the bottleneck
    reaches 1024; the decoder mirrors this back down to 64 before a 1x1 convolution maps to a
    single output channel.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_channels: int = 64, bilinear: bool = False):
        super().__init__()
        c = base_channels

        # --- Encoder (contracting path) -------------------------------------------------
        self.inc = DoubleConv(in_channels, c)        # 224 -> 224, c
        self.down1 = Down(c, c * 2)                    # 224 -> 112
        self.down2 = Down(c * 2, c * 4)                # 112 -> 56
        self.down3 = Down(c * 4, c * 8)                # 56  -> 28
        self.down4 = Down(c * 8, c * 16)              # 28  -> 14 (bottleneck)

        # --- Decoder (expanding path) ---------------------------------------------------
        self.up1 = Up(c * 16, c * 8, bilinear)        # 14  -> 28,  skip = down3
        self.up2 = Up(c * 8, c * 4, bilinear)         # 28  -> 56,  skip = down2
        self.up3 = Up(c * 4, c * 2, bilinear)         # 56  -> 112, skip = down1
        self.up4 = Up(c * 2, c, bilinear)             # 112 -> 224, skip = inc

        # 1x1 convolution to the segmentation logits.
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: keep each level's output to use later as a skip connection.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck

        # Decoder: upsample and fuse with the matching encoder features.
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
