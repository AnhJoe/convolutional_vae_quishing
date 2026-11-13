"""Image models for 3_image_vae.ipynb: the convolutional VAE and the supervised reference.

Input: the (n, 65, 65) uint8 grids of src/loaders.py (`cached_grids`), 1 for a dark module, 0 for a light one. The grids
are held on the device bit-packed (9 bytes per row instead of 65), so the whole benign train set fits in GPU memory.

The VAE is a Bernoulli convolutional VAE (65 -> 33 -> 17 -> 9 -> 5 and back). Its scores, all higher = more malicious
and all per module (divided by 65 * 65, which does not change any ranking):

    recon_error   reconstruction negative log-likelihood at the posterior mean
    neg_elbo      negative ELBO: reconstruction NLL averaged over posterior samples, plus the KL term
    kl            KL term of the posterior from the prior (in nats, not divided)
    mahalanobis   distance of the posterior mean from a Gaussian fitted to the benign train posterior means

The supervised reference is the VAE encoder plus one linear layer, trained with labels, scored as prob_malicious.

Typical use:

    train = to_device(cached_grids(RAW_DIR, benign_train, PROC_DIR / "grids_benign_train.npz"))
    vae, history = fit_image_vae(train, to_device(val_benign_grids), latent=32, beta=1.0, seed=0)
    scores = vae_scores(vae, to_device(val_grids), latent_gaussian=fit_latent_gaussian(vae, train))
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.loaders import CANVAS

N_MODULES = CANVAS * CANVAS
_SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------------
# Grids on the device
# ----------------------------------------------------------------------------------------------------------------------
class DeviceGrids:
    """Bit-packed grids on a device. `batch(idx)` returns float tensors of shape (len(idx), 1, 65, 65)."""

    def __init__(self, grids: np.ndarray, dev=None):
        if grids.ndim != 3 or grids.shape[1:] != (CANVAS, CANVAS):
            raise ValueError(f"expected grids of shape (n, {CANVAS}, {CANVAS}), got {grids.shape}")
        self.dev = dev or device()
        self.packed = torch.from_numpy(np.packbits(grids.astype(bool), axis=2)).to(self.dev)

    def __len__(self):
        return self.packed.shape[0]

    def batch(self, idx) -> torch.Tensor:
        p = self.packed[torch.as_tensor(idx, device=self.dev)]
        bits = (p.unsqueeze(-1) >> _SHIFTS.to(self.dev)) & 1
        return bits.flatten(2)[..., :CANVAS].unsqueeze(1).float()


    def subset(self, idx) -> "DeviceGrids":
        """The grids at positions idx, sharing the device."""
        out = object.__new__(DeviceGrids)
        out.dev, out.packed = self.dev, self.packed[torch.as_tensor(np.asarray(idx), device=self.dev)]
        return out


def to_device(grids: np.ndarray, dev=None) -> DeviceGrids:
    return DeviceGrids(grids, dev)


def concat(*parts: DeviceGrids) -> DeviceGrids:
    """Grids of several sets in one, in the order given (for the reference: benign train, then malicious train)."""
    out = object.__new__(DeviceGrids)
    out.dev, out.packed = parts[0].dev, torch.cat([p.packed for p in parts])
    return out


def _batches(n, size, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


# ----------------------------------------------------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------------------------------------------------
def _down(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


def _up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))


class ConvEncoder(nn.Module):
    """1 x 65 x 65 -> 256 x 5 x 5, flattened."""
    channels = (32, 64, 128, 256)
    side = 5

    def __init__(self):
        super().__init__()
        c = (1,) + self.channels
        self.net = nn.Sequential(*[_down(a, b) for a, b in zip(c[:-1], c[1:])])
        self.out_dim = self.channels[-1] * self.side ** 2

    def forward(self, x):
        return self.net(x).flatten(1)


class ImageVAE(nn.Module):
    def __init__(self, latent: int = 32):
        super().__init__()
        self.latent = latent
        self.encoder = ConvEncoder()
        self.posterior = nn.Linear(self.encoder.out_dim, 2 * latent)
        ch = ConvEncoder.channels[::-1]
        self.from_z = nn.Linear(latent, self.encoder.out_dim)
        self.decoder = nn.Sequential(*[_up(a, b) for a, b in zip(ch[:-1], ch[1:])],
                                     nn.ConvTranspose2d(ch[-1], 1, 3, stride=2, padding=1))    # 5 -> 9 -> 17 -> 33 -> 65

    def encode(self, x):
        mean, logvar = self.posterior(self.encoder(x)).chunk(2, dim=1)
        return mean, logvar.clamp(-12, 8)

    def decode(self, z):
        return self.decoder(self.from_z(z).view(-1, ConvEncoder.channels[-1], ConvEncoder.side, ConvEncoder.side))

    @staticmethod
    def kl_per_dim(mean, logvar):
        return 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar)

    @staticmethod
    def nll(logits, x):
        return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)

    def forward(self, x):
        """Per item: reconstruction NLL from one posterior sample (summed over modules) and KL per latent dimension."""
        mean, logvar = self.encode(x)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        return self.nll(self.decode(z), x), self.kl_per_dim(mean, logvar)


class ImageReference(nn.Module):
    """Supervised reference: the VAE encoder plus one linear layer. Output is the malicious logit."""

    def __init__(self):
        super().__init__()
        self.encoder = ConvEncoder()
        self.linear = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, x):
        return self.linear(self.encoder(x)).squeeze(1)


# ----------------------------------------------------------------------------------------------------------------------
# VAE: training and scores
# ----------------------------------------------------------------------------------------------------------------------
def fit_image_vae(train: DeviceGrids, val_benign: DeviceGrids, latent: int = 32, beta: float = 1.0, seed: int = 0,
                  epochs: int = 40, batch: int = 256, lr: float = 1e-3, patience: int = 5, kl_warmup: int = 0,
                  verbose: bool = True) -> tuple[ImageVAE, pd.DataFrame]:
    """Train on benign train grids, early-stop on the validation benign negative ELBO (needs no labels).

    Returns the best model and one history row per epoch: training reconstruction and KL terms, the effective beta
    (with `kl_warmup` epochs of linear warm-up), active latent units (mean KL per dimension above 0.01, the posterior
    collapse diagnostic) and the validation benign negative ELBO and reconstruction error per module.
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = ImageVAE(latent).to(train.dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best, best_state, bad = [], np.inf, None, 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        b = beta * min(1.0, (epoch + 1) / kl_warmup) if kl_warmup else beta
        rec_sum, kl_dims, n = 0.0, torch.zeros(latent, device=train.dev), 0
        for idx in _batches(len(train), batch, rng):
            x = train.batch(idx)
            rec, kl = model(x)
            loss = (rec + b * kl.sum(1)).mean() / N_MODULES
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            rec_sum += rec.sum().item()
            kl_dims += kl.detach().sum(0)
            n += len(idx)
        kl_dims = (kl_dims / n).cpu().numpy()
        val = vae_scores(model, val_benign, n_samples=1, seed=seed)
        h = {"epoch": epoch, "beta": b, "train_recon_per_module": rec_sum / n / N_MODULES, "train_kl": float(kl_dims.sum()),
             "active_units": int((kl_dims > 0.01).sum()), "val_benign_neg_elbo": float(val["neg_elbo"].mean()),
             "val_benign_recon_error": float(val["recon_error"].mean()), "seconds": time.time() - t0}
        history.append(h)
        if verbose:
            print(f"epoch {epoch:2d}  recon/module {h['train_recon_per_module']:.4f}  KL {h['train_kl']:.2f}  "
                  f"active {h['active_units']}/{latent}  val -ELBO/module {h['val_benign_neg_elbo']:.4f}  ({h['seconds']:.0f}s)")
        if h["val_benign_neg_elbo"] < best - 1e-5:
            best, best_state, bad = h["val_benign_neg_elbo"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


@torch.no_grad()
def encode_means(model: ImageVAE, grids: DeviceGrids, batch: int = 1024) -> np.ndarray:
    model.eval()
    return np.concatenate([model.encode(grids.batch(i))[0].cpu().numpy() for i in _batches(len(grids), batch)])


def fit_latent_gaussian(model: ImageVAE, benign_train: DeviceGrids, ridge: float = 1e-6) -> dict:
    """Mean and precision of the benign train posterior means, for the `mahalanobis` score."""
    mu = encode_means(model, benign_train).astype(np.float64)
    cov = np.cov(mu, rowvar=False) + ridge * np.eye(mu.shape[1])
    return {"mean": mu.mean(axis=0), "precision": np.linalg.inv(cov)}


@torch.no_grad()
def vae_scores(model: ImageVAE, grids: DeviceGrids, latent_gaussian: dict | None = None, n_samples: int = 5,
               seed: int = 0, batch: int = 512) -> dict[str, np.ndarray]:
    """recon_error, neg_elbo and kl for every grid, plus mahalanobis when `latent_gaussian` is given (see the top)."""
    model.eval()
    g = torch.Generator(device=grids.dev).manual_seed(seed)
    out = {"recon_error": [], "neg_elbo": [], "kl": [], "mean": []}
    for idx in _batches(len(grids), batch):
        x = grids.batch(idx)
        mean, logvar = model.encode(x)
        kl = model.kl_per_dim(mean, logvar).sum(1)
        std = (0.5 * logvar).exp()
        rec = sum(model.nll(model.decode(mean + torch.randn(mean.shape, generator=g, device=mean.device) * std), x)
                  for _ in range(n_samples)) / n_samples
        out["recon_error"].append((model.nll(model.decode(mean), x) / N_MODULES).double().cpu().numpy())
        out["neg_elbo"].append(((rec + kl) / N_MODULES).double().cpu().numpy())
        out["kl"].append(kl.double().cpu().numpy())
        out["mean"].append(mean.double().cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    mu = res.pop("mean")
    if latent_gaussian is not None:
        d = mu - latent_gaussian["mean"]
        res["mahalanobis"] = np.sqrt(np.einsum("ij,jk,ik->i", d, latent_gaussian["precision"], d))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Supervised reference
# ----------------------------------------------------------------------------------------------------------------------
def _auroc(y, s):
    from src.evaluate import auroc
    return auroc(y, s)


def fit_image_reference(train: DeviceGrids, train_labels, val: DeviceGrids, val_labels, seed: int = 0, epochs: int = 20,
                        batch: int = 512, lr: float = 1e-3, patience: int = 3, verbose: bool = True
                        ) -> tuple[ImageReference, pd.DataFrame]:
    """Train on benign and malicious train grids with binary cross-entropy, early-stop on validation AUROC."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    y = torch.tensor(np.asarray(train_labels, dtype=np.float32), device=train.dev)
    model = ImageReference().to(train.dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, best, best_state, bad = [], -np.inf, None, 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total = 0.0
        for idx in _batches(len(train), batch, rng):
            loss = F.binary_cross_entropy_with_logits(model(train.batch(idx)), y[torch.as_tensor(idx, device=train.dev)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        va = _auroc(val_labels, reference_logits(model, val))
        history.append({"epoch": epoch, "train_bce": total / len(train), "val_auroc": va, "seconds": time.time() - t0})
        if verbose:
            print(f"epoch {epoch:2d}  bce {total / len(train):.4f}  val AUROC {va:.4f}  ({time.time() - t0:.0f}s)")
        if va > best + 1e-5:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


@torch.no_grad()
def reference_logits(model: ImageReference, grids: DeviceGrids, batch: int = 1024) -> np.ndarray:
    model.eval()
    return np.concatenate([model(grids.batch(i)).double().cpu().numpy() for i in _batches(len(grids), batch)])


def reference_scores(model: ImageReference, grids: DeviceGrids) -> np.ndarray:
    """prob_malicious, computed in float64 so that confident items do not all round to exactly 1."""
    return 1.0 / (1.0 + np.exp(-reference_logits(model, grids)))
