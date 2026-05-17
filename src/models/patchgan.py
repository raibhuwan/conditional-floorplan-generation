import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for semantic floor plan generation.

    It receives:
      condition: outline + room count channel  [B, 2, H, W]
      mask: semantic layout representation     [B, 9, H, W]

    It outputs:
      patch-level real/fake logits             [B, 1, h, w]
    """

    def __init__(self, condition_channels=2, mask_channels=9, base=32):
        super().__init__()

        in_channels = condition_channels + mask_channels

        def block(in_ch, out_ch, use_bn=True):
            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base, use_bn=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, condition, mask):
        x = torch.cat([condition, mask], dim=1)
        return self.net(x)