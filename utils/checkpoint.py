from pathlib import Path
import torch


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    # Replace only after writing succeeds, keeping the previous checkpoint intact.
    torch.save(state, temporary)
    temporary.replace(path)


def load_decoder(decoder, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'decoder' in state and state.get('format_version') != 2:
        raise ValueError('Unsupported checkpoint format; expected version 2')
    # Accept either a resumable checkpoint or a standalone decoder state dict.
    decoder.load_state_dict(state['decoder'] if 'decoder' in state else state, strict=True)
