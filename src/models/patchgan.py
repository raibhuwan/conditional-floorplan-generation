import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for conditional semantic floor-plan generation.

    Inputs:
        condition:
            2 channels consisting of a filled binary support mask and a
            normalised encoded connected-region count channel.

        mask:
            9-channel semantic representation of either the ground-truth
            layout or the generated layout.

    Output:
        A spatial grid of real/fake logits representing local patch-level
        discriminator decisions.
    """

    # Initialise the discriminator blocks and final patch-level output layer.
    def __init__(self, condition_channels=2, mask_channels=9, base=32):
        super().__init__()

        # Combine the conditional input channels with the semantic mask channels.
        in_channels = condition_channels + mask_channels

        # Build one discriminator block using convolution, optional batch
        # normalisation and LeakyReLU activation.
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

            # Batch normalisation is applied to intermediate discriminator blocks
            # but can be disabled for the first block.
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Progressively reduce spatial resolution while increasing feature depth.
        # The final convolution returns a grid of local real/fake logits.
        self.net = nn.Sequential(
            *block(in_channels, base, use_bn=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    # Concatenate the condition and semantic representation before
    # evaluating the pair with the discriminator.
    def forward(self, condition, mask):
        x = torch.cat([condition, mask], dim=1)

        # Return raw patch-level logits for adversarial loss calculation.
        return self.net(x)