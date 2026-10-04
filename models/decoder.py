"""Decoder g: maps relu4_1 features back to an RGB image.

Roughly mirrors the encoder: pooling is replaced by nearest upsampling, and there is
no normalization layer (it would wash out the style statistics AdaIN just injected).
The last conv has no activation; its raw output is clamped only when saving images.
"""
from torch import nn

# Output channels of each 3x3 conv; upsampling (x2) follows the convs at these indices.
CHANNELS = [256, 256, 256, 256, 128, 128, 64, 64, 3]
UPSAMPLE_AFTER = {0, 4, 6}


def build_decoder():
    # Layer order fixes the state_dict keys of saved decoders; keep it unchanged.
    layers = []
    in_channels = 512
    for index, out_channels in enumerate(CHANNELS):
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, out_channels, 3)]
        if index < len(CHANNELS) - 1:
            layers.append(nn.ReLU())
        if index in UPSAMPLE_AFTER:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        in_channels = out_channels
    return nn.Sequential(*layers)
