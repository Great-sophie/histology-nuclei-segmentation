import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        base=16,
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            base,
        )

        self.enc2 = DoubleConv(
            base,
            base * 2,
        )

        self.enc3 = DoubleConv(
            base * 2,
            base * 4,
        )

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            base * 4,
            base * 8,
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base * 8,
            base * 4,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv(
            base * 8,
            base * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv(
            base * 4,
            base * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(
            base * 2,
            base,
        )

        self.head = nn.Conv2d(
            base,
            out_channels,
            kernel_size=1,
        )

    @staticmethod
    def _match(skip, x):
        if skip.shape[-2:] == x.shape[-2:]:
            return skip

        return F.interpolate(
            skip,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)

        s2 = self.enc2(
            self.pool(s1)
        )

        s3 = self.enc3(
            self.pool(s2)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(s3)
        )

        # Decoder 3
        x = self.up3(b)

        x = torch.cat(
            [
                x,
                self._match(s3, x),
            ],
            dim=1,
        )

        x = self.dec3(x)

        # Decoder 2
        x = self.up2(x)

        x = torch.cat(
            [
                x,
                self._match(s2, x),
            ],
            dim=1,
        )

        x = self.dec2(x)

        # Decoder 1
        x = self.up1(x)

        x = torch.cat(
            [
                x,
                self._match(s1, x),
            ],
            dim=1,
        )

        x = self.dec1(x)

        # Binary segmentation logits
        return self.head(x)
