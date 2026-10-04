"""Adaptive Instance Normalization (paper Eq. 8).

    AdaIN(x, y) = sigma(y) * (x - mu(x)) / sigma(x) + mu(y)

Statistics are taken per sample and per channel over spatial positions, so a style
is fully described by its channel-wise (mean, std); its feature map is never needed.
"""
import torch


def calc_mean_std(features, eps=1e-5):
    """Channel-wise (mean, std) of NCHW features, each shaped (N, C, 1, 1)."""
    if features.ndim != 4:
        raise ValueError(f'Expected NCHW features, got shape {tuple(features.shape)}')
    # Population variance as in the paper (no Bessel correction); eps keeps std > 0.
    variance, mean = torch.var_mean(features, dim=(2, 3), correction=0, keepdim=True)
    return mean, (variance + eps).sqrt()


def adain(content, style_mean, style_std):
    """Normalize content features, then re-scale and shift them with the style statistics."""
    content_mean, content_std = calc_mean_std(content)
    return (content - content_mean) / content_std * style_std + style_mean
