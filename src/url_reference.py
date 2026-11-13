"""URL supervised reference for 4_url_vae.ipynb: the URL team's GRU encoder plus one linear layer.

The architecture and training settings are those of `URLGRU` in src/url_models.py and src/run_url.py on the
`codex/url-vae-adithya-atharva` branch, so the reference has the same encoder as the URL VAE built there:

    encoder     character embedding (32) -> one-layer GRU (hidden 64), padding skipped (packed sequences),
                the last hidden state is the URL representation
    head        one linear layer -> malicious logit
    training    Adam, lr 1e-3, batch 256, 8 epochs, gradient norm clipped at 1.0, batches drawn within length bands,
                the epoch with the lowest validation binary cross-entropy is kept

Two things differ, so that it runs on the shared pipeline: the rows and splits come from the items table, and the
character ids come from `src.loaders.encode_urls` (97 ids: padding, unknown, 95 printable ASCII) instead of that branch's
98 ids (which add a start token the encoder does not use). When the URL VAE moves onto the shared pipeline, replace
`GRUEncoder` here with an import of its encoder so the two stay identical.

Saved as model `url_supervised`, score `prob_malicious` (computed in float64 from the logit, so confident items do not
round to exactly 1 and tie).

    ref, history = fit_url_reference(train_urls, train_labels, val_urls, val_labels, seed=0)
    p_val = url_reference_scores(ref, val_urls)
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.evaluate import auroc
from src.loaders import PAD_ID, VOCAB_SIZE, encode_urls

LENGTH_BANDS = [16, 32, 48, 64, 96, 128, 192]        # as LengthBatches on the URL VAE branch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class GRUEncoder(nn.Module):
    """Same layers as `Encoder` in src/url_models.py (codex/url-vae-adithya-atharva)."""

    def __init__(self, embedding: int = 32, hidden: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding, padding_idx=PAD_ID)
        self.gru = nn.GRU(embedding, hidden, batch_first=True)

    def forward(self, tokens, lengths):
        packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
        return self.gru(packed)[1][-1]


class URLReference(nn.Module):
    """Same layers as `URLGRU` in src/url_models.py: the encoder plus one linear layer, output is the malicious logit."""

    def __init__(self, embedding: int = 32, hidden: int = 64):
        super().__init__()
        self.encoder = GRUEncoder(embedding, hidden)
        self.logit = nn.Linear(hidden, 1)

    def forward(self, tokens, lengths):
        return self.logit(self.encoder(tokens, lengths)).squeeze(-1)


class DeviceURLs:
    """Encoded URLs on a device. `batch(idx)` returns (tokens cut to the longest URL in the batch, lengths)."""

    def __init__(self, urls, dev=None):
        self.dev = dev or device()
        ids = encode_urls(urls)
        self.lengths = (ids != PAD_ID).sum(axis=1)
        if (self.lengths == 0).any():
            raise ValueError("every URL needs at least one character")
        self.ids = torch.from_numpy(ids.astype(np.uint8)).to(self.dev)
        self.len_t = torch.from_numpy(self.lengths).to(self.dev)

    def __len__(self):
        return len(self.lengths)

    def batch(self, idx):
        idx_t = torch.as_tensor(np.asarray(idx), device=self.dev)
        n = int(self.lengths[np.asarray(idx)].max())
        return self.ids[idx_t, :n].long(), self.len_t[idx_t]


def _banded_batches(lengths, size, rng):
    """Shuffle within length bands, cut into batches, shuffle the batches (LengthBatches on the URL VAE branch)."""
    bands = np.searchsorted(LENGTH_BANDS, lengths)
    batches = []
    for b in range(len(LENGTH_BANDS) + 1):
        group = rng.permutation(np.flatnonzero(bands == b))
        batches += [group[i:i + size] for i in range(0, len(group), size)]
    rng.shuffle(batches)
    return batches


def _ordered_batches(lengths, size):
    order = np.argsort(lengths, kind="stable")          # similar lengths together, so little padding is processed
    return [order[i:i + size] for i in range(0, len(order), size)]


@torch.no_grad()
def url_reference_logits(model: URLReference, urls: DeviceURLs, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = np.empty(len(urls))
    for idx in _ordered_batches(urls.lengths, batch):
        out[idx] = model(*urls.batch(idx)).double().cpu().numpy()
    return out


def url_reference_scores(model: URLReference, urls: DeviceURLs) -> np.ndarray:
    """prob_malicious in float64."""
    return 1.0 / (1.0 + np.exp(-url_reference_logits(model, urls)))


def fit_url_reference(train: DeviceURLs, train_labels, val: DeviceURLs, val_labels, seed: int = 0, epochs: int = 8,
                      batch: int = 256, lr: float = 1e-3, embedding: int = 32, hidden: int = 64, verbose: bool = True
                      ) -> tuple[URLReference, pd.DataFrame]:
    """Train on benign and malicious train URLs with binary cross-entropy. Keeps the epoch with the lowest validation
    BCE, as the URL VAE branch does, and reports validation AUROC per epoch alongside."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    y = torch.tensor(np.asarray(train_labels, dtype=np.float32), device=train.dev)
    y_val = np.asarray(val_labels, dtype=np.float64)
    model = URLReference(embedding, hidden).to(train.dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best, best_state = [], np.inf, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for idx in _banded_batches(train.lengths, batch, rng):
            loss = F.binary_cross_entropy_with_logits(model(*train.batch(idx)), y[torch.as_tensor(idx, device=train.dev)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            total += loss.item() * len(idx)
        logits = url_reference_logits(model, val)
        val_bce = float(np.mean(np.logaddexp(0, logits) - y_val * logits))      # BCE with logits, computed stably
        h = {"epoch": epoch, "train_bce": total / len(train), "val_bce": val_bce, "val_auroc": auroc(y_val, logits),
             "seconds": time.time() - t0}
        history.append(h)
        if verbose:
            print(f"epoch {epoch}  bce {h['train_bce']:.4f}  val bce {val_bce:.4f}  val AUROC {h['val_auroc']:.4f}  "
                  f"({h['seconds']:.0f}s)")
        if val_bce < best:
            best, best_state = val_bce, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)
