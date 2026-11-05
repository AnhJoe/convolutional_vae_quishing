"""The URL GRU models and benign-only structural floors."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from .url_data import BOS, PAD, VOCAB_SIZE, encode


class Encoder(nn.Module):
    def __init__(self, embedding, hidden):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding, padding_idx=PAD)
        self.gru = nn.GRU(embedding, hidden, batch_first=True)

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths, batch_first=True,
                                      enforce_sorted=False)
        return self.gru(packed)[1][-1]


class URLVAE(nn.Module):
    def __init__(self, embedding, hidden, latent):
        super().__init__()
        self.encoder = Encoder(embedding, hidden)
        self.posterior = nn.Linear(hidden, 2 * latent)
        self.decoder_embedding = nn.Embedding(VOCAB_SIZE, embedding, padding_idx=PAD)
        self.decoder = nn.GRU(embedding + latent, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, tokens, lengths, generator=None, samples=1):
        mean, logvar = self.posterior(self.encoder(tokens, lengths)).chunk(2, dim=-1)
        logvar = logvar.clamp(-12, 12)
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=-1)
        previous = F.pad(tokens[:, :-1], (1, 0), value=BOS)
        context = self.decoder_embedding(previous)
        nll = torch.zeros_like(kl)
        for _ in range(samples):
            noise = torch.randn(mean.shape, generator=generator, device=mean.device)
            z = mean + noise * (0.5 * logvar).exp()
            decoder_input = torch.cat((context, z[:, None].expand(-1, tokens.size(1), -1)), dim=-1)
            logits = self.readout(self.decoder(decoder_input)[0])
            nll += F.cross_entropy(logits.transpose(1, 2), tokens, ignore_index=PAD,
                                   reduction="none").sum(dim=1)
        characters = lengths.to(tokens.device)
        return nll / samples, kl, characters


class URLGRU(nn.Module):
    def __init__(self, embedding, hidden):
        super().__init__()
        self.encoder = Encoder(embedding, hidden)
        self.logit = nn.Linear(hidden, 1)

    def forward(self, tokens, lengths):
        return self.logit(self.encoder(tokens, lengths)).squeeze(-1)


class LengthBatches:
    """Shuffle items within length bands and shuffle batches each epoch."""

    def __init__(self, dataset, batch_size, seed):
        lengths = np.fromiter((len(tokens) for tokens, _ in dataset), dtype=int)
        bands = np.searchsorted([16, 32, 48, 64, 96, 128, 192], lengths)
        self.groups = [np.flatnonzero(bands == band) for band in range(8)]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        batches = []
        for group in self.groups:
            shuffled = self.rng.permutation(group)
            batches.extend(shuffled[i:i + self.batch_size]
                           for i in range(0, len(shuffled), self.batch_size))
        self.rng.shuffle(batches)
        yield from batches


def loader(frame, batch_size, seed=None):
    # Store each sequence in bytes to keep the million-row cohort compact.
    dataset = [(bytes(encode(url)), label) for url, label in zip(frame.url, frame.label)]
    if seed is not None:
        return DataLoader(dataset, batch_sampler=LengthBatches(dataset, batch_size, seed),
                          collate_fn=collate)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


def collate(batch):
    encoded, labels = zip(*batch)
    tokens = [torch.from_numpy(np.frombuffer(row, dtype=np.uint8).astype(np.int64)) for row in encoded]
    lengths = torch.tensor([len(row) for row in tokens])
    return pad_sequence(tokens, batch_first=True, padding_value=PAD), lengths, torch.tensor(labels)


class Floors:
    def __init__(self, benign_features):
        self.mean = benign_features.mean(axis=0)
        self.scale = benign_features.std(axis=0)
        # Preserve finite distances when a benign feature has no variance.
        self.scale[self.scale == 0] = 1.0
        normalized = (benign_features - self.mean) / self.scale
        covariance = normalized.T @ normalized / len(normalized)
        self.precision = np.linalg.inv(covariance + 1e-6 * np.eye(len(self.mean)))

    def scores(self, features):
        centered = (features - self.mean) / self.scale
        squared = np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        return np.abs(centered[:, 0]), np.sqrt(np.maximum(squared, 0))
