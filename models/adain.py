"""Paper Eq. (8): independent statistics for each sample and channel."""
import torch


def calc_mean_std(features, eps=1e-5):
    if features.ndim != 4 or features.shape[-2] * features.shape[-1] < 2:
        raise ValueError('Expected NCHW features with at least two spatial elements')
    # Population variance (paper Eq. 6); explicitly avoid torch.var's default correction.
    variance, mean = torch.var_mean(features, dim=(2, 3), correction=0, keepdim=True)
    return mean, (variance + eps).sqrt()


def adaptive_instance_normalization(content, style):
    if content.shape[:2] != style.shape[:2]:
        raise ValueError('Content and style must have matching batch and channel dimensions')
    return adaptive_instance_normalization_from_stats(content, calc_mean_std(style))


def adaptive_instance_normalization_from_stats(content, style_stats):
    """Apply reusable style statistics without keeping its spatial feature map."""
    style_mean, style_std = style_stats
    expected = (*content.shape[:2], 1, 1)
    if style_mean.shape != expected or style_std.shape != expected:
        raise ValueError('Style statistics must match content batch and channel dimensions')
    content_mean, content_std = calc_mean_std(content)
    # Remove content statistics, then apply style statistics through broadcasting.
    return (content - content_mean) / content_std * style_std + style_mean
