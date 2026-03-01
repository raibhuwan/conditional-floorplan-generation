import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Simple UNet for semantic segmentation.
    in_channels=2 (outline + count_channel)
    out_channels=9 (merged semantic classes)
    """
    def __init__(self, in_channels=2, out_channels=9, base=32):
        super().__init__()

        self.down1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)           # [B, base, H, W]
        p1 = self.pool1(d1)

        d2 = self.down2(p1)          # [B, base*2, H/2, W/2]
        p2 = self.pool2(d2)

        d3 = self.down3(p2)          # [B, base*4, H/4, W/4]
        p3 = self.pool3(d3)

        d4 = self.down4(p3)          # [B, base*8, H/8, W/8]
        p4 = self.pool4(d4)

        bn = self.bottleneck(p4)     # [B, base*16, H/16, W/16]

        u4 = self.up4(bn)
        x4 = torch.cat([u4, d4], dim=1)
        x4 = self.conv4(x4)

        u3 = self.up3(x4)
        x3 = torch.cat([u3, d3], dim=1)
        x3 = self.conv3(x3)

        u2 = self.up2(x3)
        x2 = torch.cat([u2, d2], dim=1)
        x2 = self.conv2(x2)

        u1 = self.up1(x2)
        x1 = torch.cat([u1, d1], dim=1)
        x1 = self.conv1(x1)

        return self.out(x1)  # logits [B, 9, H, W]