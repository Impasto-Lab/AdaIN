"""Encoder f: frozen VGG-19 truncated at relu4_1.

Uses the "normalised" VGG-19 of the original Torch implementation. Its leading 1x1
conv folds in VGG's input preprocessing, so the encoder takes plain [0, 1] RGB.
Reflection padding replaces zero padding to reduce border artifacts.
"""
from pathlib import Path

import torch
from torch import nn

from .adain import calc_mean_std

DEFAULT_VGG_PATH = Path(__file__).resolve().parent / 'vgg_normalised.pth'
VGG_URLS = [
    'https://huggingface.co/Curise/FreestyleRet/resolve/main/vgg_normalised.pth',
    'https://hf-mirror.com/Curise/FreestyleRet/resolve/main/vgg_normalised.pth',
]

# VGG-19 up to relu4_1: conv output channels, 'M' = 2x2 max pooling.
VGG_CONFIG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512]
# Indices into the Sequential below whose outputs are used by the losses.
LAYERS = {3: 'relu1_1', 10: 'relu2_1', 17: 'relu3_1', 30: 'relu4_1'}


def build_vgg_features():
    # Indices must match the keys in vgg_normalised.pth ('0.weight', '2.weight', ...).
    layers = [nn.Conv2d(3, 3, 1)]
    in_channels = 3
    for value in VGG_CONFIG:
        if value == 'M':
            # ceil_mode keeps partial windows, so odd sizes are not silently cropped.
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            layers += [nn.ReflectionPad2d(1), nn.Conv2d(in_channels, value, 3), nn.ReLU()]
            in_channels = value
    return nn.Sequential(*layers)


def download_vgg(path):
    """Fetch the normalized VGG-19 weights, trying each mirror in turn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'VGG weights not found at {path}; downloading...')
    for url in VGG_URLS:
        try:
            # Writes to a temporary file first, so a failed download leaves nothing behind.
            torch.hub.download_url_to_file(url, str(path), progress=True)
            return
        except Exception as error:
            print(f'Download failed from {url}: {error}')
    raise RuntimeError(f'Could not download VGG weights to {path}')


class Encoder(nn.Module):
    def __init__(self, vgg_path=DEFAULT_VGG_PATH):
        super().__init__()
        self.features = build_vgg_features()
        vgg_path = Path(vgg_path)
        if not vgg_path.is_file():
            download_vgg(vgg_path)
        state = torch.load(vgg_path, map_location='cpu', weights_only=True)
        # The file holds the full VGG-19; keep only the layers up to relu4_1.
        depth = len(self.features)
        self.features.load_state_dict({k: v for k, v in state.items() if int(k.split('.')[0]) < depth})
        # Weights stay fixed, but gradients still flow through activations to the decoder.
        self.requires_grad_(False)
        self.eval()

    def forward(self, image, intermediate=False, statistics=False):
        """Encode an NCHW batch of [0, 1] RGB images.

        intermediate: return {name: value} for relu1_1..relu4_1 instead of relu4_1 only.
        statistics:   reduce each feature map to its (mean, std) right away, so large
                      maps can be freed early (all that is needed from style images).
        """
        outputs = {}
        for index, layer in enumerate(self.features):
            image = layer(image)
            if intermediate and index in LAYERS:
                outputs[LAYERS[index]] = calc_mean_std(image) if statistics else image
        if intermediate:
            return outputs
        return calc_mean_std(image) if statistics else image
