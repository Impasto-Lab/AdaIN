import torch
from torch import nn
from .adain import adaptive_instance_normalization_from_stats
from .decoder import build_decoder
from .encoder import Encoder


class AdaINNetwork(nn.Module):
    def __init__(self, vgg_path=None):
        super().__init__()
        self.encoder = Encoder(vgg_path)
        self.decoder = build_decoder()

    @torch.no_grad()
    def encode_style(self, style, intermediate=False):
        return self.encoder(style, intermediate=intermediate, statistics=True)

    @staticmethod
    @torch.no_grad()
    def make_target(content_features, style_stats, alpha=1.0):
        """Build a detached decoder target from encoded content and style statistics."""
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1')
        if alpha == 0:
            return content_features.detach()
        target = adaptive_instance_normalization_from_stats(content_features, style_stats)
        # Avoid allocating two products and a sum for the usual alpha=1 case.
        if alpha == 1:
            return target
        # Blend features, not output pixels, to control stylization strength.
        return alpha * target + (1 - alpha) * content_features

    def forward(self, content, style, alpha=1.0, training_outputs=False, style_loss=True):
        """Training returns output features, target and per-layer style statistics."""
        if not 0 <= alpha <= 1:
            raise ValueError('alpha must be between 0 and 1')
        intermediate = training_outputs and style_loss
        with torch.no_grad():
            # Targets are fixed; only the decoder receives parameter gradients.
            style_stats = self.encode_style(style, intermediate=intermediate) if alpha or intermediate else None
            style_target = style_stats['relu4_1'] if intermediate else style_stats
            content_features = self.encoder(content)
            target = self.make_target(content_features, style_target, alpha)
        output = self.decoder(target)
        if training_outputs:
            # Frozen parameters still allow d(loss)/d(output) to reach the decoder.
            features = self.encoder(output, intermediate=intermediate)
            if not intermediate:
                features = {'relu4_1': features}
            return features, target, style_stats if intermediate else {}
        return output
