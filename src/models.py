"""Networks: the image VAE, and the supervised reference CNNs (image and URL).

The supervised references reuse the encoder of the matching VAE, so the gap between the VAE and its reference is about
the training signal (labels or not), not the architecture. The URL VAE should take `CharEncoder` as its encoder for the
same reason.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import CANVAS, PAD_ID, URL_MAX_LEN, VOCAB_SIZE


# ----------------------------------------------------------------------------------------------------------------------
# Image (65 x 65 binary module grid)
# ----------------------------------------------------------------------------------------------------------------------
def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


def _deconv(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


class ConvEncoder(nn.Module):
    """1 x 65 x 65 -> 256 x 5 x 5 (65 -> 33 -> 17 -> 9 -> 5), flattened."""
    channels = (32, 64, 128, 256)
    out_side = 5

    def __init__(self):
        super().__init__()
        c = (1,) + self.channels
        self.net = nn.Sequential(*[_conv(a, b) for a, b in zip(c[:-1], c[1:])])
        self.out_dim = self.channels[-1] * self.out_side ** 2

    def forward(self, x):
        return self.net(x).flatten(1)


class ImageVAE(nn.Module):
    """Convolutional VAE with a Bernoulli likelihood over the modules of the canvas."""

    def __init__(self, latent: int = 32, beta: float = 1.0):
        super().__init__()
        self.latent, self.beta = latent, beta
        self.enc = ConvEncoder()
        self.to_stats = nn.Linear(self.enc.out_dim, 2 * latent)
        ch = ConvEncoder.channels[::-1]
        self.from_z = nn.Linear(latent, self.enc.out_dim)
        self.dec = nn.Sequential(*[_deconv(a, b) for a, b in zip(ch[:-1], ch[1:])],
                                 nn.ConvTranspose2d(ch[-1], 1, 3, stride=2, padding=1))      # 5 -> 9 -> 17 -> 33 -> 65

    def encode(self, x):
        mu, logvar = self.to_stats(self.enc(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-12, 8)

    def decode(self, z):
        h = self.from_z(z).view(-1, ConvEncoder.channels[-1], ConvEncoder.out_side, ConvEncoder.out_side)
        return self.dec(h)

    @staticmethod
    def kl_per_dim(mu, logvar):
        return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)

    def forward(self, x):
        """Training loss terms, summed over modules / latent dims, per item: (recon NLL, KL per dim)."""
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        logits = self.decode(z)
        nll = F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
        return nll, self.kl_per_dim(mu, logvar)

    @torch.no_grad()
    def score(self, x, n_samples: int = 5, generator=None) -> dict:
        """Per-module scores (divide by CANVAS**2): reconstruction NLL at the posterior mean, and the negative ELBO
        averaged over `n_samples` posterior draws. Also returns the total KL (nats) per item."""
        d = CANVAS * CANVAS
        mu, logvar = self.encode(x)
        nll_mean = F.binary_cross_entropy_with_logits(self.decode(mu), x, reduction="none").flatten(1).sum(1)
        kl = self.kl_per_dim(mu, logvar).sum(1)
        rec = torch.zeros_like(kl)
        std = (0.5 * logvar).exp()
        for _ in range(n_samples):
            eps = torch.randn(mu.shape, generator=generator, device=mu.device) if generator is not None else torch.randn_like(mu)
            rec += F.binary_cross_entropy_with_logits(self.decode(mu + eps * std), x, reduction="none").flatten(1).sum(1)
        rec /= n_samples
        return {"nll": nll_mean / d, "neg_elbo": (rec + kl) / d, "kl": kl}


class ImageCNN(nn.Module):
    """Supervised image reference: the VAE encoder plus a linear head, output is a malicious logit."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.enc = ConvEncoder()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.out_dim, 1))

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(1)


# ----------------------------------------------------------------------------------------------------------------------
# URL (character ids, length URL_MAX_LEN)
# ----------------------------------------------------------------------------------------------------------------------
class CharEncoder(nn.Module):
    """Character CNN: embedding, three conv blocks that halve the length, masked global max and mean pooling."""

    def __init__(self, emb: int = 32, channels=(64, 128, 256)):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, emb, padding_idx=PAD_ID)
        c = (emb,) + tuple(channels)
        self.convs = nn.ModuleList(
            nn.Sequential(nn.Conv1d(a, b, 5, padding=2), nn.BatchNorm1d(b), nn.LeakyReLU(0.2),
                          nn.Conv1d(b, b, 3, stride=2, padding=1), nn.BatchNorm1d(b), nn.LeakyReLU(0.2))
            for a, b in zip(c[:-1], c[1:]))
        self.out_dim = 2 * channels[-1]
        self.out_len = URL_MAX_LEN // 2 ** len(channels)

    def forward(self, ids):
        mask = (ids != PAD_ID).float().unsqueeze(1)                    # B x 1 x L
        h = self.emb(ids).transpose(1, 2)
        for conv in self.convs:
            h = conv(h)
            mask = F.max_pool1d(mask, 2, 2)
        h_max = h.masked_fill(mask == 0, -1e4).amax(2)
        h_mean = (h * mask).sum(2) / mask.sum(2).clamp(min=1)
        return torch.cat([h_max, h_mean], 1)


class CharCNN(nn.Module):
    """Supervised URL reference: CharEncoder plus a linear head, output is a malicious logit."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.enc = CharEncoder()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.enc.out_dim, 1))

    def forward(self, ids):
        return self.head(self.enc(ids)).squeeze(1)
