"""VGG-19 encoder with reflection padding and normalized weights (up to relu4_1)."""
from pathlib import Path
import torch
from torch import nn
from .adain import calc_mean_std

DEFAULT_VGG_PATH = Path(__file__).resolve().parent.parent / 'models' / 'vgg_normalised.pth'
VGG_URLS = [
    'https://huggingface.co/Curise/FreestyleRet/resolve/main/vgg_normalised.pth',
    'https://hf-mirror.com/Curise/FreestyleRet/resolve/main/vgg_normalised.pth',
]


def ensure_vgg(vgg_path):
    vgg_path = Path(vgg_path)
    if vgg_path.is_file():
        return vgg_path
    vgg_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"VGG weights not found at {vgg_path}. Downloading normalized VGG-19...")
    last_err = None
    for url in VGG_URLS:
        try:
            torch.hub.download_url_to_file(url, str(vgg_path), progress=True)
            if vgg_path.is_file() and vgg_path.stat().st_size > 0:
                print(f"Downloaded VGG weights to {vgg_path}")
                return vgg_path
        except Exception as e:
            last_err = e
            print(f"Failed to download from {url}: {e}")
    raise RuntimeError(f"Could not download VGG weights to {vgg_path}: {last_err}")


def build_vgg_features():
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # 3: relu1_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # 6: relu1_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # 10: relu2_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # 13: relu2_2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # 17: relu3_1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # 20: relu3_2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # 23: relu3_3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # 26: relu3_4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # 30: relu4_1
    )


class Encoder(nn.Module):
    endpoints = {3: 'relu1_1', 10: 'relu2_1', 17: 'relu3_1', 30: 'relu4_1'}

    def __init__(self, vgg_path=None):
        super().__init__()
        self.features = build_vgg_features()
        if vgg_path is not False:
            vgg_path = Path(vgg_path) if vgg_path is not None else DEFAULT_VGG_PATH
            if not vgg_path.is_file():
                vgg_path = ensure_vgg(vgg_path)
            state_dict = torch.load(vgg_path, map_location='cpu', weights_only=True)
            sub_sd = {k: v for k, v in state_dict.items() if int(k.split('.')[0]) <= 30}
            self.features.load_state_dict(sub_sd)
        self.requires_grad_(False)
        self.eval()

    def forward(self, image, intermediate=False, statistics=False):
        """Optionally reduce features immediately to channel means/stds."""
        result = {}
        for index, layer in enumerate(self.features):
            image = layer(image)
            if intermediate and index in self.endpoints:
                result[self.endpoints[index]] = calc_mean_std(image) if statistics else image
        if intermediate:
            return result
        return calc_mean_std(image) if statistics else image
