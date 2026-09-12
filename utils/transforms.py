import torch
from PIL import Image
from .image import to_tensor


def resize_short_side(image, size):
    if size == 0:
        return image
    if size < 0:
        raise ValueError('Image size cannot be negative')
    width, height = image.size
    if min(width, height) == size:
        return image
    scale = size / min(width, height)
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)


def training_transform(image, resize_size, crop_size):
    # Resize preserves aspect ratio; independent crops form unpaired batches.
    image = resize_short_side(image, resize_size)
    width, height = image.size
    if min(width, height) < crop_size:
        raise ValueError('Resized image is smaller than crop_size')
    left = torch.randint(width - crop_size + 1, ()).item()
    top = torch.randint(height - crop_size + 1, ()).item()
    return to_tensor(image.crop((left, top, left + crop_size, top + crop_size)))


def inference_transform(image, size):
    image = resize_short_side(image, size)
    if min(image.size) < 16:
        raise ValueError('Inference image dimensions must be at least 16 pixels')
    return to_tensor(image)
