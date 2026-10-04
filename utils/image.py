"""Image discovery, loading and saving."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def image_paths(root):
    """A single image file, or all images under a directory (recursive, sorted)."""
    root = Path(root)
    if root.is_file() and root.suffix.lower() in EXTENSIONS:
        return [root]
    paths = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found: {root}')
    return paths


def load_rgb(path):
    with Image.open(path) as image:
        # Apply the EXIF orientation first, so phone photos are not stylized sideways.
        return ImageOps.exif_transpose(image).convert('RGB')


def to_tensor(image):
    """HWC uint8 PIL image -> CHW float tensor in [0, 1] (VGG preprocessing lives in the encoder)."""
    return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255


def save_image(tensor, path):
    """Save a CHW tensor; out-of-range decoder output is clamped only here, at export."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = tensor.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(path)
