import argparse
import csv
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.adain import calc_mean_std
from models.network import AdaINNetwork
from utils.checkpoint import atomic_save
from utils.config import ROOT, load_config, project_path, resolve_device
from utils.dataset import ImageDataset, ReplacementSampler, split_paths
from utils.image import load_rgb, to_tensor, save_image
from utils.reproducibility import seed_everything, random_state, restore_random_state
from utils.transforms import resize_short_side


def perceptual_losses(output_features, target, style_stats):
    """Mean squared errors, equally weighted style layers; not raw L2 norms."""
    # Reconstruct the AdaIN target, rather than the original content features.
    content_loss = F.mse_loss(output_features['relu4_1'], target)
    style_loss = content_loss.new_zeros(())
    for name, (target_mean, target_std) in style_stats.items():
        mean, std = calc_mean_std(output_features[name])
        style_loss = style_loss + F.mse_loss(mean, target_mean) + F.mse_loss(std, target_std)
    return content_loss, style_loss


def validation_image(path, resize_size, crop_size):
    """Use a fixed center crop matching training scale for comparable previews."""
    image = resize_short_side(load_rgb(path), resize_size)
    width, height = image.size
    left, top = (width - crop_size) // 2, (height - crop_size) // 2
    return to_tensor(image.crop((left, top, left + crop_size, top + crop_size)))


def main():
    # Read the config first so its keys also define command-line overrides.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=str(ROOT / 'configs/train.json'))
    known, _ = bootstrap.parse_known_args()
    config = load_config(known.config)
    parser = argparse.ArgumentParser(description=__doc__, parents=[bootstrap])
    parser.add_argument('--resume', help='Full latest.pt training checkpoint')
    for key, value in config.items():
        parser.add_argument('--' + key.replace('_', '-'), type=type(value), default=None)
    args = vars(parser.parse_args())
    config.update({key: value for key, value in args.items() if key in config and value is not None})
    resume = args['resume']

    positive = ['batch_size', 'resize_size', 'crop_size', 'max_steps', 'learning_rate',
                'log_every', 'save_every', 'preview_every']
    if any(config[key] <= 0 for key in positive):
        raise ValueError('Batch/size/step/learning-rate/interval values must be positive')
    if config['num_workers'] < 0 or config['style_weight'] < 0 or config['learning_rate_decay'] < 0:
        raise ValueError('Worker count, style weight and learning rate decay cannot be negative')
    if config['crop_size'] < 16 or config['crop_size'] % 8 or config['resize_size'] < config['crop_size']:
        raise ValueError('crop_size must be >=16, divisible by 8, and <= resize_size')
    seed_everything(config['seed'])
    device = resolve_device(config['device'])
    output_dir = project_path(config['output_dir'])
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose --output-dir or use --resume')
    vgg_path = project_path(config['vgg_path']) if 'vgg_path' in config and config['vgg_path'] else None
    model = AdaINNetwork(vgg_path=vgg_path).to(device)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config['learning_rate'])
    start = 0
    checkpoint = None
    if resume:
        checkpoint = torch.load(project_path(resume), map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') != 2:
            raise ValueError('Unsupported checkpoint format; expected version 2')
        # Keep the training setup fixed while allowing a longer run or new device.
        mutable = {'max_steps', 'output_dir', 'device', 'log_every', 'save_every', 'preview_every'}
        for key, value in checkpoint['config'].items():
            if key not in mutable and config[key] != value:
                raise ValueError(f'Resume config differs for {key}; reuse the saved config')
        model.decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        if start >= config['max_steps']:
            raise ValueError('max_steps must exceed the checkpoint step')
    splits = {}
    loaders = []
    for kind in ('content', 'style'):
        # Independent loaders pair unrelated content and style images.
        training, validation = split_paths(config[f'{kind}_dir'], config['validation_fraction'], config['seed'])
        splits[kind] = {'train': [str(p) for p in training], 'val': [str(p) for p in validation]}
        dataset = ImageDataset(training, config['resize_size'], config['crop_size'])
        sampler = ReplacementSampler(len(dataset), (config['max_steps'] - start) * config['batch_size'])
        # DataLoader worker seeds use a separate generator from sampling/cropping.
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
    preview_content = validation_image(splits['content']['val'][0], config['resize_size'], config['crop_size']).unsqueeze(0).to(device)
    preview_style = validation_image(splits['style']['val'][0], config['resize_size'], config['crop_size']).unsqueeze(0).to(device)
    # The frozen encoder and validation pair never change during training.
    with torch.no_grad():
        preview_target = model.make_target(model.encoder(preview_content), model.encode_style(preview_style))
    if checkpoint:
        # Restore after model initialization, which consumes random numbers.
        restore_random_state(checkpoint['rng'])
    del checkpoint  # Release the loaded CPU copies of decoder/optimizer state.
    model.train()
    model.encoder.eval()
    log_path = output_dir / 'losses.csv'
    with log_path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if stream.tell() == 0:
            writer.writerow(['step', 'loss', 'content', 'style', 'learning_rate'])
        for step, (content, style) in enumerate(zip(*loaders), start + 1):
            # Compute LR from the absolute step so resumed runs keep the schedule.
            lr = config['learning_rate'] / (1 + config['learning_rate_decay'] * (step - 1))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            features, target, style_stats = model(
                content.to(device, non_blocking=True), style.to(device, non_blocking=True),
                training_outputs=True, style_loss=config['style_weight'] > 0)
            content_loss, style_loss = perceptual_losses(features, target, style_stats)
            loss = content_loss + config['style_weight'] * style_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            # The optimizer contains decoder parameters only.
            optimizer.step()
            if step == start + 1 or step % config['log_every'] == 0 or step == config['max_steps']:
                values = [step, loss.item(), content_loss.item(), style_loss.item(), lr]
                writer.writerow(values)
                stream.flush()
                print(f'step {step}: loss={values[1]:.5f}, content={values[2]:.5f}, style={values[3]:.5f}', flush=True)
            if step % config['preview_every'] == 0 or step == config['max_steps']:
                # These convolution/ReLU/pooling layers have no train/eval behavior.
                with torch.no_grad():
                    preview = model.decoder(preview_target)
                save_image(torch.cat([preview_content[0], preview_style[0], preview[0]], dim=2),
                           output_dir / 'previews' / f'{step:06d}.jpg')
            if step % config['save_every'] == 0 or step == config['max_steps']:
                # Full state resumes training; the small decoder file is for sharing.
                atomic_save({'format_version': 2, 'step': step, 'decoder': model.decoder.state_dict(),
                             'optimizer': optimizer.state_dict(), 'config': config, 'splits': splits,
                             'rng': random_state()}, output_dir / 'latest.pt')
                atomic_save(model.decoder.state_dict(), output_dir / f'decoder_{step:06d}.pth')
    print(f'Training complete: {output_dir}')


if __name__ == '__main__':
    main()
