"""Config loading, project-relative paths and device selection."""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        return json.load(stream)


def project_path(path):
    """Resolve relative paths against the project root, so scripts work from any cwd."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def resolve_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)
