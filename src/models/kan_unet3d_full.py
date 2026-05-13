"""
    3D U-Net + KAN bottleneck - controlled ablation variant.

    Identical to UNet3D (Cicek et al. 2016) in every way except the bottleneck:
    the two LeakyReLU activations in the bottleneck DoubleConv are replaced with
    KAN activations (Bernstein + B-spline, same as kan_unet3d.py).

    Purpose: isolate whether KAN activations themselves drive any improvement,
    independent of the lighter architecture used in KANUNet3d.
    One variable changed, everything else identical.

    All encoder and decoder layers are unchanged - same DoubleConv3d, same
    InstanceNorm, same LeakyReLU, same features, same skip connections.
"""
import torch
import torch.nn as nn

# reusing building blocks from the other model files rather than reimplementing
from .unet3d import DoubleConv3d
from .kan_unet3d import BernsteinActivation, BSplineActivation


class _KANConvBlock(nn.Module):
    """
        Conv3d -> InstanceNorm -> KAB -> KAS (drop-in replacement for Conv+LeakyReLU).
        Underscore prefix because this is an internal building block, not part of the public API.
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.kab = BernsteinActivation(out_ch, order=4)
        self.kas = BSplineActivation(out_ch, grid_size=5, spline_order=3)

    def _kan(self, x: torch.Tensor) -> torch.Tensor:
        # same reshape trick as KABSBlock3d._apply_kan in kan_unet3d.py
        # flatten to (N, C), run KAN in fp32, reshape back to (B, C, D, H, W)
        B, C, D, H, W = x.shape
        flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C).float()
        flat = self.kab(flat)
        flat = self.kas(flat)
        return flat.to(x.dtype).reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._kan(self.norm(self.conv(x)))


class _KANDoubleConv(nn.Module):
    """
        Two _KANConvBlocks stacked - same interface as DoubleConv3d so it's a drop-in.
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _KANConvBlock(in_ch, out_ch),
            _KANConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KANUNet3dFull(nn.Module):
    """
        3D U-Net with KAN activations at the bottleneck only.
        Every layer is identical to UNet3D except the bottleneck DoubleConv,
        where LeakyReLU is replaced by Bernstein + B-spline KAN activations.
        Encoder, decoder, skip connections, and head are all unchanged.

        ~22.59M parameters, same scale as the 3D U-Net baseline (22.58M).
        The tiny difference is the extra spline parameters in the bottleneck.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv3d(ch, f))
            ch = f

        # only difference from UNet3D: KAN activations replace LeakyReLU here
        self.bottleneck = _KANDoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upsamples.append(nn.ConvTranspose3d(f * 2, f, 2, stride=2))
            self.decoders.append(DoubleConv3d(f * 2, f))

        self.head = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)
