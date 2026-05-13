"""
    3D U-KABS - adaptation of the hybrid KAN U-Net for volumetric medical image segmentation.

    Original 2D paper: 'A hybrid Kolmogorov-Arnold network for medical image segmentation' arXiv:2602.07702 (Feb 2026)

    3D adaptation for BraTS2021:
    - 2D convolutions to 3D, bilinear upsampling to trilinear, 2D max pool to 3D
    - ViT-style tokenization replaced with 1x1x1 channel projection (appropriate at small 3D spatial dims)
    - KAN layers (Bernstein + B-spline) placed at the two deepest encoder/decoder levels (16^3 and 8^3),
    where spatial resolution is small enough to make per-voxel KAN computation feasible on 8GB VRAM

    Architecture:
    - Encoder: ConvSE x3 -> KABS (KAN) x2
    - Decoder: KABS (KAN) x1 -> ConvSE x3
    - Head: 1x1x1 conv -> 3 classes (NCR, ED, ET)
"""
from math import comb

import torch
import torch.nn as nn
import torch.nn.functional as F


# Building blocks
class SEBlock3d(nn.Module):
    """
        3D Squeeze-and-Excitation: learns per-channel importance weights.
        Squeezes spatial dims to a single value per channel, then uses a small
        FC network to produce a weight for each channel. Channels the model
        thinks are important get amplified, less useful ones get suppressed.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # floor at 4 so the bottleneck doesn't get too small for low channel counts
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)
        return x * w


class ConvSEBlock3d(nn.Module):
    """
        Single conv block with SE attention (one encoder/decoder level in the CNN part).
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # BatchNorm here (vs InstanceNorm in unet3d) following the original U-KABS paper
            # at batch size 1 this is unconventional but worked fine in practice
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


# KAN activation layers
class BernsteinActivation(nn.Module):
    """
        KAB: per-channel Bernstein polynomial activation (globally smooth).
        for more info: https://en.wikipedia.org/wiki/Bernstein_polynomial

        Standard networks have fixed activations (ReLU, GELU) and learnable weights.
        KANs flip this - the activation function itself is learnable.
        Bernstein polynomials are smooth everywhere, good for capturing broad structure.

        phi(z) = w_b * SiLU(z) + w_s * sum_r theta_r * C(R,r) * z_norm^r * (1-z_norm)^(R-r)

        z_norm is a per-batch min-max normalisation to [0,1] for numerical stability.
        theta, w_b, w_s are all learnable. w_b and w_s start at 1 so initially acts like SiLU.
    """
    def __init__(self, channels: int, order: int = 4) -> None:
        super().__init__()
        self.order = order
        self.theta = nn.Parameter(torch.zeros(channels, order + 1))
        self.w_b = nn.Parameter(torch.ones(channels))
        self.w_s = nn.Parameter(torch.ones(channels))

        binom = torch.tensor([comb(order, r) for r in range(order + 1)], dtype=torch.float32)
        r_idx = torch.arange(order + 1, dtype=torch.float32)
        # register_buffer: not learnable but gets moved to the right device with .to(device)
        self.register_buffer("binom", binom)
        self.register_buffer("r_idx", r_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C) already cast to float32 by KABSBlock3d._apply_kan
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        z = ((x - x_min) / (x_max - x_min + 1e-6)).clamp(0.0, 1.0)

        z3 = z.unsqueeze(-1)
        r = self.r_idx.view(1, 1, -1)
        basis = self.binom * (z3 ** r) * ((1.0 - z3) ** (self.order - r))

        poly = (basis * self.theta.unsqueeze(0)).sum(-1)
        return self.w_b * F.silu(x) + self.w_s * poly


class BSplineActivation(nn.Module):
    """
        KAS: per-channel B-spline activation (locally adaptive).

        Complements Bernstein: where Bernstein is globally smooth, B-splines are
        locally supported so they can fit fine-grained boundary features independently.

        phi(x) = w_b * SiLU(x) + w_s * sum_i c_i * B_i(x)

        B_i are B-spline basis functions over a uniform grid on [-1, 1].
        c_i, w_b, w_s are learnable.
    """
    def __init__(self, channels: int, grid_size: int = 5, spline_order: int = 3) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        n_basis = grid_size + spline_order

        self.coeff = nn.Parameter(torch.zeros(channels, n_basis))
        self.w_b = nn.Parameter(torch.ones(channels))
        self.w_s = nn.Parameter(torch.ones(channels))

        # extended knot vector: extra knots beyond each boundary for numerical stability
        k = spline_order
        h = 2.0 / grid_size
        grid = torch.linspace(-1.0 - k * h, 1.0 + k * h, grid_size + 2 * k + 1)
        self.register_buffer("grid", grid)

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """
            Cox-de Boor recursion
            x: (N, C) in (-1,1) -> (N, C, G+k).
        """
        G, k = self.grid_size, self.spline_order
        g  = self.grid.view(1, 1, -1)
        x3 = x.unsqueeze(-1)

        # order-0: indicator function, 1 where grid[i] <= x < grid[i+1]
        b = ((x3 >= g[..., :-1]) & (x3 < g[..., 1:])).to(x.dtype)

        for p in range(1, k + 1):
            n = G + 2 * k - p
            t0 = g[..., :n]
            tp = g[..., p : n + p]
            tp1 = g[..., p + 1 : n + p + 1]
            t1 = g[..., 1 : n + 1]

            ld = tp  - t0
            rd = tp1 - t1
            left = torch.where(ld.abs() > 1e-8, (x3 - t0) / ld, torch.zeros_like(x3))
            right = torch.where(rd.abs() > 1e-8, (tp1 - x3) / rd, torch.zeros_like(x3))
            b = left * b[..., :-1] + right * b[..., 1:]

        return b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)                              # squash to (-1, 1) for the grid
        basis = self._basis(x_norm)
        spline = (basis * self.coeff.unsqueeze(0)).sum(-1)
        return self.w_b * F.silu(x) + self.w_s * spline


# KABS block (KAN encoder/decoder level)
class KABSBlock3d(nn.Module):
    """
        3D KABS block: project -> SE -> KAB -> KAS -> DwConv3d -> LN + residual.
        KAN activations run in fp32 regardless of AMP context.
        Bernstein involves x^4 which loses precision in fp16 and destabilises training.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 1x1x1 proj only needed when channel dims differ
        self.proj = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.se = SEBlock3d(out_channels)
        self.kab = BernsteinActivation(out_channels, order=4)
        self.kas = BSplineActivation(out_channels, grid_size=5, spline_order=3)
        # depth-wise conv: each channel convolved independently (groups=channels)
        # much cheaper than a full conv, keeps the block lightweight
        self.dw = nn.Conv3d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.ln = nn.LayerNorm(out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def _apply_kan(self, x: torch.Tensor) -> torch.Tensor:
        """
            Flatten (B,C,D,H,W) to (N,C), run KAN activations in fp32, reshape back.
        """
        B, C, D, H, W = x.shape
        # KAN activations are per-channel so we treat each voxel as an independent sample
        flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C).float()
        flat = self.kab(flat)
        flat = self.kas(flat)
        return flat.to(x.dtype).reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        z = self.proj(x)
        z = self.se(z)
        z = self._apply_kan(z)
        z = self.dw(z)
        # LayerNorm expects channels last, so permute, norm, permute back
        z = self.ln(z.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return z + residual


# Full model
class KANUNet3d(nn.Module):
    """
        3D U-KABS for BraTS2021.

        CNN blocks handle the shallow levels where spatial resolution is high.
        KAN blocks only at the two deepest levels (16^3 and 8^3) where the spatial
        size is small enough that per-voxel KAN computation is affordable.
        
        Encoder:
            enc1 ConvSE 128^3 -> f1
            enc2 ConvSE 64^3 -> f2
            enc3 ConvSE 32^3 -> f3
            enc4 KABS 16^3 -> f4 (KAN starts here)
            bot KABS 8^3 -> f4 (bottleneck)

        Decoder:
            dec4 KABS 16^3 -> f4
            dec3 ConvSE 32^3 -> f3
            dec2 ConvSE 64^3 -> f2
            dec1 ConvSE 128^3 -> f1
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: tuple = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        f1, f2, f3, f4 = features
        self.pool = nn.MaxPool3d(2)

        self.enc1 = ConvSEBlock3d(in_channels, f1)
        self.enc2 = ConvSEBlock3d(f1, f2)
        self.enc3 = ConvSEBlock3d(f2, f3)
        self.enc4 = KABSBlock3d(f3, f4)
        self.bottleneck = KABSBlock3d(f4, f4)

        # trilinear Upsample instead of ConvTranspose3d, simpler and works well here
        # align_corners=False is the recommended default for trilinear
        self.up4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec4 = KABSBlock3d(f4 + f4, f4) # f4+f4 because skip connection doubles channels

        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec3 = ConvSEBlock3d(f4 + f3, f3)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec2 = ConvSEBlock3d(f3 + f2, f2)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec1 = ConvSEBlock3d(f2 + f1, f1)

        self.head = nn.Conv3d(f1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3)) # KAN
        x  = self.bottleneck(self.pool(s4)) # KAN

        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.head(x)
