"""
    Attention U-Net (Oktay et al. 2018, arXiv:1804.03999)

    Same as UNet3D except each skip connection passes through an attention gate
    before being concatenated into the decoder. The gate learns to suppress
    background activations and focus on tumour regions, trained purely from
    the segmentation loss with no extra supervision.
"""
import torch
import torch.nn as nn

from .unet3d import DoubleConv3d


class AttentionGate3d(nn.Module):
    """
        Soft attention gate for a skip connection.

        Takes two inputs:
        - g: gating signal from the decoder (knows what we're looking for)
        - x: skip features from the encoder (the features we want to filter)

        Produces a spatial weight map alpha in [0,1]. Multiplying x by alpha
        suppresses irrelevant regions and amplifies tumour-like ones.
        The gate is learned entirely from the segmentation loss.
    """
    def __init__(self, F_g: int, F_x: int, F_int: int) -> None:
        super().__init__()
        # project both inputs to the same intermediate size before combining
        self.Wg = nn.Conv3d(F_g, F_int, kernel_size=1, bias=True)
        self.Wx = nn.Conv3d(F_x, F_int, kernel_size=1, bias=True)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True), # collapse to single attention map
            nn.Sigmoid(), # output in [0,1]
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # add projected gating signal and skip, then pass through sigmoid to get alpha
        alpha = self.psi(self.relu(self.Wg(g) + self.Wx(x))) # (B, 1, H, W, D)
        return x * alpha # broadcast alpha across all channels


class AttentionUNet3D(nn.Module):
    """
        Attention U-Net: standard U-Net with attention gates on every skip connection.

        in_channels: number of MRI modalities (4 for BraTS)
        out_channels: number of segmentation classes (3 for BraTS: NCR, ED, ET)
        features: channel sizes at each encoder level
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
        self.upsamples = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv3d(ch, f))
            ch = f

        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.upsamples.append(nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2))
            # F_int = f//2 following the original paper
            self.attention.append(AttentionGate3d(F_g=f, F_x=f, F_int=f // 2))
            # attended skip (f) + upsampled (f) = f*2 channels into decoder
            self.decoders.append(DoubleConv3d(f * 2, f))

        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, att, dec, skip in zip(
            self.upsamples, self.attention, self.decoders, reversed(skips)
        ):
            x = up(x)
            skip = att(g=x, x=skip) # gate filters the skip before concatenation
            x = torch.cat([skip, x], dim=1)
            x = dec(x) 

        return self.head(x)
