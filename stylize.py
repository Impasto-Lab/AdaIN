"""Stylize every content image with every style image using a trained AdaIN decoder."""
import argparse
import hashlib

import torch
from torch.nn import functional as F

from models.network import AdaINNetwork
from utils.checkpoint import load_decoder
from utils.config import project_path, resolve_device
from utils.image import image_paths, load_rgb, save_image
from utils.transforms import inference_transform


def load_batch(path, size, device):
    """Image file -> (1, 3, H, W) tensor on `device`."""
    return inference_transform(load_rgb(path), size).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--content', required=True, help='Image or directory')
    parser.add_argument('--style', required=True, help='Image or directory')
    parser.add_argument('--decoder', default='models/decoder.pth', help='Decoder .pth or full training .pt')
    parser.add_argument('--vgg', default='models/vgg_normalised.pth', help='Path to vgg_normalised.pth')
    parser.add_argument('--output-dir', default='outputs/stylized')
    parser.add_argument('--content-size', type=int, default=512, help='Short side; 0 keeps size')
    parser.add_argument('--style-size', type=int, default=512, help='Short side; 0 keeps size')
    parser.add_argument('--alpha', type=float, default=1.0, help='Stylization strength in [0, 1]')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        parser.error('alpha must be between 0 and 1')
    if args.content_size < 0 or args.style_size < 0:
        parser.error('image sizes must be nonnegative')
    # Validate inputs before building the encoder, which may download VGG weights.
    contents = image_paths(args.content)
    styles = image_paths(args.style)
    decoder_path = project_path(args.decoder)
    if not decoder_path.is_file():
        parser.error(f'Decoder not found: {decoder_path}')

    device = resolve_device(args.device)
    model = AdaINNetwork(project_path(args.vgg))
    load_decoder(model.decoder, decoder_path)
    model.to(device).eval()
    output_dir = project_path(args.output_dir)

    with torch.inference_mode():
        # A style is fully described by its relu4_1 (mean, std): 2 x 512 numbers, cheap to cache.
        style_stats = [model.encode_style(load_batch(path, args.style_size, device)) for path in styles]
        for content_path in contents:
            content = load_batch(content_path, args.content_size, device)
            height, width = content.shape[-2:]
            # Pad to a multiple of 8 (three 2x poolings) so the output aligns pixel-for-pixel
            # with the input; the padding is cropped off again below.
            content = F.pad(content, (0, -width % 8, 0, -height % 8), mode='reflect')
            # Encode each content image once and reuse it for every style.
            content_features = model.encoder(content)
            for style_path, stats in zip(styles, style_stats):
                output = model.decoder(model.make_target(content_features, stats, args.alpha))
                # Hash of both full paths keeps same-named files from different folders apart.
                tag = hashlib.sha256(f'{content_path.resolve()}|{style_path.resolve()}'.encode()).hexdigest()[:8]
                path = output_dir / f'{content_path.stem}_{style_path.stem}_{tag}.png'
                save_image(output[0, :, :height, :width], path)
                print(path)


if __name__ == '__main__':
    main()
