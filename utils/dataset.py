import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .image import image_paths, load_rgb
from .transforms import training_transform


def split_paths(root, validation_fraction=0.05, seed=42):
    """Stable filename split; no copies or writes to the source dataset."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Dataset must be a directory: {root}')
    paths = image_paths(root)
    if not 0 < validation_fraction < 1 or len(paths) < 2:
        raise ValueError('Need >=2 images and validation_fraction in (0, 1)')
    paths.sort(key=lambda p: hashlib.sha256(f'{seed}:{p.relative_to(root).as_posix()}'.encode()).digest())
    count = min(len(paths) - 1, max(1, round(len(paths) * validation_fraction)))
    return paths[count:], paths[:count]


class ImageDataset(Dataset):
    def __init__(self, paths, resize_size=512, crop_size=256):
        self.paths = paths
        self.resize_size, self.crop_size = resize_size, crop_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            return training_transform(load_rgb(path), self.resize_size, self.crop_size)
        except Exception as error:
            raise RuntimeError(f'Cannot load training image: {path}') from error


class ReplacementSampler(torch.utils.data.Sampler):
    """Independent sampling with replacement, bounded by remaining training steps."""
    def __init__(self, size, count):
        self.size, self.count = size, count

    def __len__(self):
        return self.count

    def __iter__(self):
        # Draw lazily so a checkpoint captures the next draw when num_workers=0.
        for _ in range(self.count):
            yield torch.randint(self.size, ()).item()
