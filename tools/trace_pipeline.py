"""Record one real AdaIN run and embed it into assets/adain-pipeline.html.

    python tools/trace_pipeline.py [--run outputs/runs/adain_normalized]

Everything the page shows comes from here: inputs, encoder/decoder activations, AdaIN
statistics, outputs for several alphas, the training log and previews, and the source
lines shown in the code panel. The data replaces the page's <script id="trace"> block.

Encodings (base64 strings):
  feature maps  uint8, value = byte / 255 * scale, one scale per map  (ReLU outputs are >= 0)
  signed maps   uint8, value = (byte - 127.5) / 127.5 * scale         (AdaIN maps can be < 0)
  images        data:image/jpeg URIs
"""
import argparse
import base64
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.adain import adain, calc_mean_std  # noqa: E402
from models.network import AdaINNetwork  # noqa: E402
from utils.checkpoint import load_decoder  # noqa: E402
from utils.image import load_rgb  # noqa: E402
from utils.transforms import inference_transform  # noqa: E402

PAGE = ROOT / 'assets' / 'adain-pipeline.html'
SOURCES = ['stylize.py', 'train.py', 'models/adain.py', 'models/encoder.py', 'models/decoder.py',
           'models/network.py', 'utils/transforms.py']
ALPHAS = [round(a / 10, 1) for a in range(11)]
PREVIEW_STEPS = [1000, 2000, 5000, 10000, 20000, 40000, 80000, 160000]
CHANNELS_SHOWN = 4      # stacked channels drawn per layer
ADAIN_CHANNELS = 6      # channels selectable in the AdaIN section
MAP_SIZE = 96           # feature maps are average-pooled down to at most this size


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def jpeg(image, size=256, quality=88):
    if isinstance(image, torch.Tensor):
        image = Image.fromarray(image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def pooled(features):
    """(C, H, W) -> (C, h, w) with max(h, w) <= MAP_SIZE."""
    h, w = features.shape[-2:]
    factor = max(1, -(-max(h, w) // MAP_SIZE))
    return F.avg_pool2d(features[None], factor)[0] if factor > 1 else features


def encode_maps(features):
    """Non-negative maps -> {'h', 'w', 's': scales, 'd': uint8 bytes}."""
    features = pooled(features).float().cpu()
    scales = features.flatten(1).quantile(0.995, dim=1).clamp_min(1e-6)
    data = (features / scales[:, None, None]).clamp(0, 1).mul(255).round().byte().numpy()
    return {'h': features.shape[1], 'w': features.shape[2], 's': [round(float(s), 4) for s in scales], 'd': b64(data)}


def encode_signed(feature, scale):
    data = (feature.float().cpu() / scale).clamp(-1, 1).mul(127.5).add(127.5).round().byte().numpy()
    return b64(data)


def top_channels(features, k):
    """Indices of the k channels with the largest spatial variance (the most visible structure)."""
    return features.flatten(1).var(dim=1).topk(k).indices.tolist()


def rounded(tensor, digits=4):
    return [round(float(v), digits) for v in tensor.flatten()]


@torch.no_grad()
def trace_encoder(model, image):
    """Output of every layer that changes shape or content, keyed like the page's node list."""
    outputs, x = [], image
    for index, layer in enumerate(model.encoder.features):
        x = layer(x)
        # Record after each 1x1 conv, ReLU and pooling layer (skip pads and pre-activation convs).
        if index == 0 or isinstance(layer, (torch.nn.ReLU, torch.nn.MaxPool2d)):
            outputs.append(x[0])
    return outputs


@torch.no_grad()
def trace_decoder(model, target):
    outputs, x = [], target
    layers = list(model.decoder)
    for index, layer in enumerate(layers):
        x = layer(x)
        last = index == len(layers) - 1
        if isinstance(layer, (torch.nn.ReLU, torch.nn.Upsample)) or last:
            outputs.append(x[0])
    return outputs


def histogram(values, low, high, bins=48):
    counts, _ = np.histogram(values.float().cpu().numpy().ravel(), bins=bins, range=(low, high))
    return [round(float(c), 5) for c in counts / counts.sum()]


def training_log(run):
    path = run / 'losses.csv'
    if not path.is_file():
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        rows = [[float(v) for v in row[:4]] for row in list(csv.reader(stream))[1:]]
    rows = np.array(rows)
    # Block means over 400 points: raw per-batch losses are too noisy to read.
    blocks = np.array_split(rows, 400)
    smooth = np.array([block.mean(axis=0) for block in blocks])
    previews, pair = [], None
    for step in PREVIEW_STEPS:
        file = run / 'previews' / f'{step:06d}.jpg'
        if not file.is_file():
            continue
        image = Image.open(file).convert('RGB')
        size = image.height  # content | style | stylized, each size x size
        if pair is None:
            pair = [jpeg(image.crop((i * size, 0, (i + 1) * size, size)), 192) for i in range(2)]
        previews.append({'step': step, 'img': jpeg(image.crop((2 * size, 0, 3 * size, size)), 192)})
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    keys = ['batch_size', 'crop_size', 'resize_size', 'learning_rate', 'learning_rate_decay', 'style_weight', 'max_steps']
    return {
        'first': rounded(torch.tensor(rows[0, 1:4])), 'last': rounded(torch.tensor(rows[-1, 1:4])),
        'step': [int(s) for s in smooth[:, 0]], 'loss': rounded(torch.tensor(smooth[:, 1])),
        'content': rounded(torch.tensor(smooth[:, 2])), 'style': rounded(torch.tensor(smooth[:, 3])),
        'previews': previews, 'pair': pair, 'config': {k: config[k] for k in keys},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--content', default='assets/content.jpg')
    parser.add_argument('--style', default='assets/style.jpg')
    parser.add_argument('--decoder', default='models/decoder.pth')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--run', default='outputs/runs/adain_normalized', help='Training run with losses.csv and previews/')
    args = parser.parse_args()

    torch.manual_seed(0)
    model = AdaINNetwork()
    load_decoder(model.decoder, ROOT / args.decoder)
    model.eval()
    content_image, style_image = load_rgb(ROOT / args.content), load_rgb(ROOT / args.style)
    content = inference_transform(content_image, args.size)[None]
    style = inference_transform(style_image, args.size)[None]
    assert content.shape[-1] % 8 == 0 and content.shape[-2] % 8 == 0, 'pick a size divisible by 8'

    with torch.no_grad():
        # --- encoder: every layer for both images, each showing its own most active channels.
        enc_c, enc_s = trace_encoder(model, content), trace_encoder(model, style)
        encoder = []
        for fc, fs in zip(enc_c, enc_s):
            if fc.shape[0] == 3:
                # The 1x1 conv output is mean-subtracted BGR (can be negative): shift it to >= 0 for display.
                maps = [encode_maps(f + 128) for f in (fc, fs)]
            else:
                maps = [encode_maps(f[top_channels(f, CHANNELS_SHOWN)]) for f in (fc, fs)]
            encoder.append({'shape': list(fc.shape), 'c': maps[0], 's': maps[1]})

        # --- statistics and AdaIN at relu4_1
        f_c, f_s = enc_c[-1][None], enc_s[-1][None]
        mean_c, std_c = calc_mean_std(f_c)
        mean_s, std_s = calc_mean_std(f_s)
        target = adain(f_c, mean_s, std_s)
        mean_t, std_t = calc_mean_std(target)
        # Channels where content and style statistics differ most make the shift visible.
        gap = ((mean_s - mean_c).abs() + (std_s - std_c).abs()).flatten()
        live = (std_c.flatten() > 0.05) & (std_s.flatten() > 0.05)
        channels = (gap * live).topk(ADAIN_CHANNELS).indices.tolist()
        adain_channels = []
        for k in channels:
            x, y, t = f_c[0, k], f_s[0, k], target[0, k]
            normalized = (x - mean_c[0, k]) / std_c[0, k]
            # t and f(s) share one colour scale and one histogram axis, so they can be compared directly.
            t_scale = float(max(t.abs().quantile(.995), y.quantile(.995)))
            low, high = float(min(t.min(), y.min())), float(max(t.quantile(.995), y.quantile(.995)))
            x_high = float(x.quantile(.995))
            adain_channels.append({
                'k': k, 'mc': round(float(mean_c[0, k]), 3), 'sc': round(float(std_c[0, k]), 3),
                'ms': round(float(mean_s[0, k]), 3), 'ss': round(float(std_s[0, k]), 3),
                'x': encode_maps(x[None]), 'y': encode_maps(y[None]),
                'n': encode_signed(normalized, 3.0), 't': encode_signed(t, t_scale), 'ys': encode_signed(y, t_scale),
                'hx': {'lo': round(float(x.min()), 4), 'hi': round(x_high, 4), 'x': histogram(x, float(x.min()), x_high)},
                'hty': {'lo': round(low, 4), 'hi': round(high, 4), 't': histogram(t, low, high), 'y': histogram(y, low, high)},
            })
        stats = {'mc': rounded(mean_c, 3), 'sc': rounded(std_c, 3), 'ms': rounded(mean_s, 3), 'ss': rounded(std_s, 3),
                 'mt': rounded(mean_t, 3), 'st': rounded(std_t, 3),
                 'err': float(max((mean_t - mean_s).abs().max(), (std_t - std_s).abs().max()))}

        # --- decoder at alpha = 1, plus final outputs for every alpha
        dec = trace_decoder(model, target)
        decoder = []
        for out in dec[:-1]:
            picks = top_channels(out, CHANNELS_SHOWN)
            decoder.append({'shape': list(out.shape), 'm': encode_maps(out[picks])})
        target_picks = top_channels(target[0], CHANNELS_SHOWN)
        t_scale = float(target[0, target_picks].abs().quantile(.995))
        decoder_input = {'shape': list(target[0].shape), 'scale': round(t_scale, 4),
                         'h': target.shape[-2], 'w': target.shape[-1],
                         'd': encode_signed(target[0, target_picks], t_scale)}
        outputs = {}
        for alpha in ALPHAS:
            out = model.decoder(model.make_target(f_c, (mean_s, std_s), alpha))[0]
            outputs[str(alpha)] = jpeg(out, 320)
        raw = dec[-1]

    trace = {
        'size': list(content.shape[-2:]),
        'content': jpeg(content_image, 320), 'style': jpeg(style_image, 320),
        'files': {'content': args.content, 'style': args.style, 'decoder': args.decoder},
        'encoder': encoder, 'stats': stats, 'adain': adain_channels,
        'decoder_input': decoder_input, 'decoder': decoder,
        'raw_range': [round(float(raw.min()), 3), round(float(raw.max()), 3)],
        'outputs': outputs, 'train': training_log(ROOT / args.run),
        'src': {name: (ROOT / name).read_text(encoding='utf-8').splitlines() for name in SOURCES},
    }
    payload = json.dumps(trace, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')
    page = PAGE.read_text(encoding='utf-8')
    pattern = re.compile(r'(<script type="application/json" id="trace">).*?(</script>)', re.S)
    assert len(pattern.findall(page)) == 1, 'trace block not found exactly once'
    page = pattern.sub(lambda m: m.group(1) + payload + m.group(2), page)
    PAGE.write_text(page, encoding='utf-8', newline='\n')
    print(f'Wrote {len(payload) / 1e6:.2f} MB of trace data to {PAGE}')


if __name__ == '__main__':
    main()
