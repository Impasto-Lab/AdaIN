"""Train the AdaIN decoder on unpaired content and style image folders.

Settings come from a JSON config (default: configs/train.json); every key can be
overridden on the command line, e.g. style_weight -> --style-weight. With --resume,
the checkpoint's saved config is used instead and only run-time settings may change.
"""
import argparse
import csv
import json

import torch
from torch.utils.data import DataLoader

from models.encoder import DEFAULT_VGG_PATH
from models.network import AdaINNetwork
from utils.checkpoint import FORMAT_VERSION, atomic_save, load_checkpoint
from utils.config import ROOT, load_config, project_path, resolve_device
from utils.dataset import ImageDataset, ReplacementSampler, split_paths
from utils.image import load_rgb, save_image
from utils.reproducibility import random_state, restore_random_state, seed_everything
from utils.transforms import center_crop_transform

# Keys that may differ from the checkpoint on resume; everything else defines the experiment.
RESUMABLE_KEYS = {'max_steps', 'output_dir', 'device', 'num_workers', 'log_every', 'save_every', 'preview_every'}


def parse_config():
    """Return (config, checkpoint or None)."""
    # Load the base config first, so its keys can define the command-line overrides.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=str(ROOT / 'configs/train.json'),
                           help='JSON config (ignored with --resume)')
    bootstrap.add_argument('--resume', help='latest.pt checkpoint to continue from')
    known, _ = bootstrap.parse_known_args()
    checkpoint = load_checkpoint(project_path(known.resume)) if known.resume else None
    config = checkpoint['config'] if checkpoint else load_config(known.config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, value in config.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value))
    overrides = {k: v for k, v in vars(parser.parse_args()).items() if k in config and v is not None}
    if checkpoint:
        locked = sorted(k for k, v in overrides.items() if v != config[k] and k not in RESUMABLE_KEYS)
        if locked:
            parser.error(f'cannot change {", ".join(locked)} when resuming')
    config.update(overrides)

    positive = ['batch_size', 'resize_size', 'crop_size', 'max_steps', 'learning_rate',
                'log_every', 'save_every', 'preview_every']
    if any(config[key] <= 0 for key in positive):
        parser.error('batch/size/step/learning-rate/interval values must be positive')
    if config['num_workers'] < 0 or config['style_weight'] < 0 or config['learning_rate_decay'] < 0:
        parser.error('num_workers, style_weight and learning_rate_decay cannot be negative')
    # Multiple of 8 keeps encoder and decoder feature sizes aligned (three 2x down/upsamplings).
    if config['crop_size'] < 16 or config['crop_size'] % 8 or config['resize_size'] < config['crop_size']:
        parser.error('crop_size must be >=16, divisible by 8, and <= resize_size')
    if checkpoint and checkpoint['step'] >= config['max_steps']:
        parser.error('max_steps must exceed the checkpoint step')
    return config, checkpoint


def main():
    config, checkpoint = parse_config()
    seed_everything(config['seed'])
    device = resolve_device(config['device'])
    output_dir = project_path(config['output_dir'])
    if not checkpoint and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose --output-dir or use --resume')

    model = AdaINNetwork(project_path(config.get('vgg_path') or DEFAULT_VGG_PATH)).to(device)
    # Only the decoder is optimized; the encoder is frozen.
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config['learning_rate'])
    start = 0
    if checkpoint:
        model.decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']

    # Content and style come from independent loaders, so every batch pairs unrelated images.
    splits, loaders = {}, []
    for kind in ('content', 'style'):
        training, validation = split_paths(project_path(config[f'{kind}_dir']),
                                           config['validation_fraction'], config['seed'])
        splits[kind] = {'train': [str(p) for p in training], 'val': [str(p) for p in validation]}
        dataset = ImageDataset(training, config['resize_size'], config['crop_size'])
        # One sample per remaining step and batch slot: the loader ends exactly at max_steps.
        sampler = ReplacementSampler(len(dataset), (config['max_steps'] - start) * config['batch_size'])
        # A dedicated generator seeds the workers without consuming the global RNG,
        # which sampling and cropping rely on for exact resumption.
        generator = torch.Generator().manual_seed(config['seed'])
        loaders.append(DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler,
                                  num_workers=config['num_workers'], pin_memory=device.type == 'cuda',
                                  generator=generator))
        print(f'{kind}: {len(training)} training, {len(validation)} validation images', flush=True)
    if checkpoint and checkpoint['splits'] != splits:
        raise ValueError('Dataset file list changed since checkpoint')

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    (output_dir / 'splits.json').write_text(json.dumps(splits, indent=2), encoding='utf-8')

    # Fixed validation pair for previews; its AdaIN target never changes, so compute it once.
    preview_content, preview_style = (
        center_crop_transform(load_rgb(splits[kind]['val'][0]), config['resize_size'], config['crop_size'])
        .unsqueeze(0).to(device) for kind in ('content', 'style'))
    preview_target = model.make_target(model.encoder(preview_content), model.encode_style(preview_style))

    if checkpoint:
        # Restore last: model construction above consumed random numbers.
        restore_random_state(checkpoint['rng'])
    del checkpoint  # free the CPU copies of decoder/optimizer state

    log_path = output_dir / 'losses.csv'
    if start and log_path.exists():
        # Steps logged after the checkpoint will run again; drop them to avoid duplicate rows.
        with log_path.open(newline='', encoding='utf-8') as stream:
            rows = [row for row in csv.reader(stream) if not row[0].isdigit() or int(row[0]) <= start]
        with log_path.open('w', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerows(rows)

    with log_path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(['step', 'loss', 'content', 'style', 'learning_rate'])
        for step, (content, style) in enumerate(zip(*loaders), start + 1):
            # Inverse-time decay from the absolute step, so resumed runs keep the schedule.
            lr = config['learning_rate'] / (1 + config['learning_rate_decay'] * (step - 1))
            for group in optimizer.param_groups:
                group['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            content_loss, style_loss = model(content.to(device, non_blocking=True),
                                             style.to(device, non_blocking=True))
            loss = content_loss + config['style_weight'] * style_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            optimizer.step()

            last = step == config['max_steps']
            if step == start + 1 or step % config['log_every'] == 0 or last:
                values = [step, loss.item(), content_loss.item(), style_loss.item(), lr]
                writer.writerow(values)
                stream.flush()
                print(f'step {step}: loss={values[1]:.5f}, content={values[2]:.5f}, style={values[3]:.5f}', flush=True)
            if step % config['preview_every'] == 0 or last:
                # Side by side: content | style | stylized.
                with torch.no_grad():
                    preview = model.decoder(preview_target)
                save_image(torch.cat([preview_content[0], preview_style[0], preview[0]], dim=2),
                           output_dir / 'previews' / f'{step:06d}.jpg')
            if step % config['save_every'] == 0 or last:
                # latest.pt holds everything needed to resume; decoder_*.pth is the small file to share.
                atomic_save({'format_version': FORMAT_VERSION, 'step': step,
                             'decoder': model.decoder.state_dict(), 'optimizer': optimizer.state_dict(),
                             'config': config, 'splits': splits, 'rng': random_state()},
                            output_dir / 'latest.pt')
                atomic_save(model.decoder.state_dict(), output_dir / f'decoder_{step:06d}.pth')
    print(f'Training complete: {output_dir}')


if __name__ == '__main__':
    main()
