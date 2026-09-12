import argparse
import hashlib

import torch
from torch.nn import functional as F

from models.network import AdaINNetwork
from utils.checkpoint import load_decoder
from utils.config import project_path, resolve_device
from utils.image import image_paths, load_rgb, save_image
from utils.transforms import inference_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--content', required=True, help='Image or directory')
    parser.add_argument('--style', required=True, help='Image or directory')
    parser.add_argument('--decoder', default='models/decoder.pth', help='Decoder .pth or full training .pt')
    parser.add_argument('--vgg', default='models/vgg_normalised.pth', help='Path to vgg_normalised.pth')
    parser.add_argument('--output-dir', default='outputs/stylized')
    parser.add_argument('--content-size', type=int, default=512, help='Short side; 0 keeps size')
    parser.add_argument('--style-size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        parser.error('alpha must be between 0 and 1')
    if args.content_size < 0 or args.style_size < 0:
        parser.error('image sizes must be nonnegative')

    # Check local inputs before initializing VGG, which may download weights.
    contents = image_paths(args.content)
    styles = image_paths(args.style)

    decoder_path = project_path(args.decoder)
    if not decoder_path.is_file():
        parser.error(f'Decoder not found: {decoder_path}')

    device = resolve_device(args.device)
    model = AdaINNetwork(vgg_path=project_path(args.vgg))
    load_decoder(model.decoder, decoder_path)
    model.to(device).eval()
    output_dir = project_path(args.output_dir)

    # Inference does not build a gradient graph.
    with torch.inference_mode():
        # Only 2 * 512 scalars per style; do not cache full feature maps.
        style_cache = []
        for style_path in styles:
            statistics = None
            if args.alpha:
                style = inference_transform(load_rgb(style_path), args.style_size).unsqueeze(0).to(device)
                statistics = model.encode_style(style)
                del style
            style_cache.append((style_path, statistics))
        for content_path in contents:
            content = inference_transform(load_rgb(content_path), args.content_size).unsqueeze(0).to(device)
            height, width = content.shape[-2:]

            # Three pooling layers require padding to a multiple of eight.
            if height % 8 or width % 8:
                content = F.pad(content, (0, (-width) % 8, 0, (-height) % 8), mode='reflect')
            content_features = model.encoder(content)
            reconstruction = model.decoder(content_features) if args.alpha == 0 else None

            for style_path, statistics in style_cache:
                output = reconstruction if reconstruction is not None else model.decoder(
                    model.make_target(content_features, statistics, args.alpha))
                output = output[0, :, :height, :width]

                # Distinguish identical filenames from different input subdirectories.
                tag = hashlib.sha256(f'{content_path.resolve()}|{style_path.resolve()}'.encode()).hexdigest()[:8]
                path = output_dir / f'{content_path.stem}_{style_path.stem}_{tag}.png'
                save_image(output, path)
                print(path)


if __name__ == '__main__':
    main()
