"""PIL image -> tensor transforms for training, previews and inference."""
import torch
from PIL import Image

from .image import to_tensor


def resize_short_side(image, size):
    """Resize so the shorter side equals `size`, keeping aspect ratio; 0 keeps the original."""
    if size < 0:
        raise ValueError('Image size cannot be negative')
    width, height = image.size
    if size == 0 or min(width, height) == size:
        return image
    scale = size / min(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)


def training_transform(image, resize_size, crop_size):
    """Resize, then take a random square crop (uses the global torch RNG, so it is seedable)."""
    image = resize_short_side(image, resize_size)
    width, height = image.size
    if min(width, height) < crop_size:
        raise ValueError('Resized image is smaller than crop_size')
    left = torch.randint(width - crop_size + 1, ()).item()
    top = torch.randint(height - crop_size + 1, ()).item()
    return to_tensor(image.crop((left, top, left + crop_size, top + crop_size)))


def center_crop_transform(image, resize_size, crop_size):
    """Deterministic counterpart of training_transform, so previews are comparable across steps."""
    image = resize_short_side(image, resize_size)
    width, height = image.size
    left, top = (width - crop_size) // 2, (height - crop_size) // 2
    return to_tensor(image.crop((left, top, left + crop_size, top + crop_size)))


def inference_transform(image, size):
    image = resize_short_side(image, size)
    # 16 px -> 2 px at relu4_1 (after three 2x poolings), the minimum a 1-px reflection pad accepts.
    if min(image.size) < 16:
        raise ValueError('Inference image dimensions must be at least 16 pixels')
    return to_tensor(image)
