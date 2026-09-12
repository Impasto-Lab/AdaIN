"""Mirror VGG through relu4_1; no normalization or output activation."""
from torch import nn


def build_decoder():
    layers = []
    incoming = 512
    for index, outgoing in enumerate([256, 256, 256, 256, 128, 128, 64, 64, 3]):
        layers.extend([nn.ReflectionPad2d(1), nn.Conv2d(incoming, outgoing, 3)])
        if outgoing != 3:
            layers.append(nn.ReLU())
        if index in (0, 4, 6):
            # Undo the encoder's three spatial downsampling operations.
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        incoming = outgoing
    return nn.Sequential(*layers)
