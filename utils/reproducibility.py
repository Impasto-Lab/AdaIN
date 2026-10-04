"""Seeding and RNG state capture for exactly resumable training."""
import torch


def seed_everything(seed):
    # All randomness (sampling, cropping, decoder init) uses torch; this also seeds CUDA.
    torch.manual_seed(seed)


def random_state():
    return {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_random_state(state):
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
