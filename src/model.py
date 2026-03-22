
import torch
import torch.nn as nn


# ── Building block: two conv layers with BN + ReLU ───────────────────────────

class DoubleConv(nn.Module):
    """
    The repeated unit inside every encoder and decoder block:
        Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU

    Why two convolutions? One conv has a 3×3 receptive field.
    Two convolutions stacked = 5×5 effective receptive field,
    capturing more context without using a bigger (slower) kernel.

    Why BatchNorm? Normalizes activations between layers — keeps
    gradient magnitudes stable so training doesn't collapse early.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ── Encoder block: DoubleConv + MaxPool ──────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    Applies DoubleConv then halves spatial dimensions via MaxPool.
    Returns BOTH the feature map (for skip connection) and the
    pooled output (to pass to the next encoder level).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)      # full resolution — saved for skip
        pooled   = self.pool(features)  # half resolution — passed down
        return features, pooled


# ── Decoder block: Upsample + concat skip + DoubleConv ───────────────────────

class DecoderBlock(nn.Module):
    """
    Upsamples the input, concatenates the matching skip connection
    from the encoder, then applies DoubleConv to fuse them.

    Why concat and not add? Concatenation preserves both sets of
    features — the high-level semantics from the decoder path AND
    the fine spatial detail from the encoder path. Addition would
    blend them, potentially losing fine boundary information.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        # Note: DoubleConv in_channels = out_channels * 2 because
        # we concat upsampled (out_channels) + skip (out_channels)

    def forward(self, x, skip):
        x = self.up(x)               # upsample: doubles spatial size

        # Handle potential size mismatch from odd input dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode='bilinear', align_corners=False
            )

        x = torch.cat([skip, x], dim=1)   # concat along channel dim
        return self.conv(x)


# ── Full U-Net ────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Full U-Net for binary segmentation.

    Architecture:
        Encoder: 4 levels, channels [64, 128, 256, 512]
        Bottleneck: 1024 channels at lowest spatial resolution
        Decoder: 4 levels mirroring encoder
        Output: 1×H×W sigmoid map (defect probability per pixel)

    Args:
        in_channels  : number of input channels (3 for RGB)
        out_channels : number of output channels (1 for binary mask)
        features     : list of channel sizes for each encoder level
    """
    def __init__(self, in_channels=3, out_channels=1,
                 features=[64, 128, 256, 512]):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(f * 2, f))

        # ── Output head ─────────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # 1×1 conv maps 64 channels → 1 channel (the mask)

    def forward(self, x):
        # ── Encoder pass — collect skip connections ──────────────────────────
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decoder pass — use skip connections in reverse order ─────────────
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # ── Output — sigmoid squashes to [0,1] probability per pixel ─────────
        return torch.sigmoid(self.output_conv(x))
