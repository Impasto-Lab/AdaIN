"""Checkpoint saving and decoder loading."""
from pathlib import Path

import torch

FORMAT_VERSION = 2


def atomic_save(state, path):
    """Write to a temporary file, then rename, so a crash never corrupts the previous file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def check_format(checkpoint):
    if checkpoint.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Unsupported checkpoint format; expected version {FORMAT_VERSION}')
    return checkpoint


def load_checkpoint(path):
    """Load a full training checkpoint (latest.pt)."""
    return check_format(torch.load(path, map_location='cpu', weights_only=True))


def load_decoder(decoder, path):
    """Load decoder weights from a standalone decoder .pth or a full training checkpoint."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'decoder' in state:
        state = check_format(state)['decoder']
    decoder.load_state_dict(state)
