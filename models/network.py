"""Full AdaIN pipeline: VGG encoder -> AdaIN -> decoder, plus training losses."""
import torch
from torch import nn
from torch.nn import functional as F

from .adain import adain, calc_mean_std
from .decoder import build_decoder
from .encoder import DEFAULT_VGG_PATH, Encoder


class AdaINNetwork(nn.Module):
    def __init__(self, vgg_path=DEFAULT_VGG_PATH):
        super().__init__()
        self.encoder = Encoder(vgg_path)  # frozen
        self.decoder = build_decoder()    # the only trainable part

    @torch.no_grad()
    def encode_style(self, style, intermediate=False):
        """relu4_1 (mean, std) of style images, or a {layer: (mean, std)} dict of all loss layers."""
        return self.encoder(style, intermediate=intermediate, statistics=True)

    @staticmethod
    @torch.no_grad()
    def make_target(content_features, style_stats, alpha=1.0):
        """Decoder input t = (1 - alpha) * f(c) + alpha * AdaIN(f(c), s).

        Blending happens in feature space, which controls stylization strength at test time.
        """
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1')
        if alpha == 0:
            return content_features
        target = adain(content_features, *style_stats)
        return target if alpha == 1 else alpha * target + (1 - alpha) * content_features

    def forward(self, content, style):
        """Training pass on unpaired batches; returns (content_loss, style_loss)."""
        with torch.no_grad():
            # The target is a constant: no gradients are needed before the decoder.
            style_stats = self.encode_style(style, intermediate=True)
            target = self.make_target(self.encoder(content), style_stats['relu4_1'])
        output_features = self.encoder(self.decoder(target), intermediate=True)

        # Content loss: match the AdaIN target, not the original content features.
        content_loss = F.mse_loss(output_features['relu4_1'], target)
        # Style loss: match channel mean/std at relu1_1..relu4_1, equally weighted.
        style_loss = 0
        for name, (style_mean, style_std) in style_stats.items():
            mean, std = calc_mean_std(output_features[name])
            style_loss = style_loss + F.mse_loss(mean, style_mean) + F.mse_loss(std, style_std)
        return content_loss, style_loss
