"""3D U-Net — Çiçek et al. 2016 (arXiv:1606.06650)"""
import torch
import torch.nn as nn


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            # InstanceNorm over BatchNorm because batch size is 1 during training
            #!BatchNorm statistics are unreliable with a single sample per batch
            nn.InstanceNorm3d(out_ch),
            # LeakyReLU instead of ReLU to avoid dead neurons (small negative slope = 0.01)
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            # bias=False because InstanceNorm has its own learnable shift parameter
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
        Encoder-decoder with skip connections.
        in_channels: number of MRI modalities (4 for BraTS: FLAIR, T1, T1CE, T2)
        out_channels: number of segmentation classes (3 for BraTS: NCR, ED, ET)
        features: channel sizes at each encoder level, doubles each level by default
    """
    def __init__(self, in_channels=4, out_channels=3, features=(32, 64, 128, 256)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv3d(ch, f))
            ch = f

        # bottleneck: features[-1]*2 = 512 channels at the coarsest spatial resolution
        self.bottleneck = DoubleConv3d(features[-1], features[-1] * 2)

        for f in reversed(features):
            # ConvTranspose3d is a learnable upsample, stride=2 doubles spatial dims
            self.upsamples.append(nn.ConvTranspose3d(f * 2, f, 2, stride=2))
            # after concatenating skip connection the channel count doubles, hence f*2 in
            self.decoders.append(DoubleConv3d(f * 2, f))

        # 1x1x1 conv (projects from feature channels down to output classes, no spatial mixing)
        self.head = nn.Conv3d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)  # save encoder output before pooling
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # concatenate skip connection (passes fine-grained spatial detail)
            # from encoder to decoder at matching resolution
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        # raw logits 
        return self.head(x)
