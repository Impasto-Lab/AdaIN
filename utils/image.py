from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def image_paths(root):
    root = Path(root)
    paths = [root] if root.is_file() and root.suffix.lower() in EXTENSIONS else sorted(
        path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found: {root}')
    return paths


def load_rgb(path):
    with Image.open(path) as image:
        # Apply camera orientation before measuring or cropping the image.
        return ImageOps.exif_transpose(image).convert('RGB')


def to_tensor(image):
    # HWC uint8 image -> CHW float RGB; VGG normalization happens in Encoder.
    return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255


def save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clamp only for export; training needs gradients through raw decoder output.
    pixels = tensor.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)
