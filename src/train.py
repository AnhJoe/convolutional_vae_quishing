"""Train a model and write its validation and test scores.

    python -m src.train image_vae --latent 32 --beta 1 --seed 0
    python -m src.train image_ref --seed 0          # supervised image CNN (ceiling for the image VAE)
    python -m src.train url_ref --seed 0            # supervised character CNN (ceiling for the URL VAE)

Each run writes
    results/scores/<run>.parquet   item_id plus one or more score columns (higher = more malicious), validation + test rows
    results/runs/<run>.json        config, per-epoch history (losses, KL per epoch, active latent units), validation AUROC
    results/checkpoints/<run>.pt   best weights

The image models use only items whose code fits the 65 x 65 canvas (fits_canvas). The VAE trains on benign train items
and early-stops on the validation benign negative ELBO, which needs no labels. The references train on benign and
malicious train items and early-stop on validation AUROC.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CANVAS, CKPT_DIR, RESULTS_DIR, SCORES_DIR
from .features import encode_urls
from .metrics import auroc
from .models import CharCNN, ImageCNN, ImageVAE
from .prepare_data import load_grids, load_items

RUNS_DIR = RESULTS_DIR / "runs"
ITEM_COLS = ["item_id", "label", "split", "url", "domain"]
IMAGE_COLS = ITEM_COLS + ["fits_canvas"]   # needs the image cache (python -m src.prepare_data)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------------------------------
# Data: everything is held on the device as compact uint8 tensors and expanded per batch
# ----------------------------------------------------------------------------------------------------------------------
_SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


class ImageData:
    def __init__(self, item_ids: np.ndarray, dev):
        self.packed = torch.from_numpy(np.ascontiguousarray(load_grids()[np.sort(item_ids)])).to(dev)
        self.index = {i: k for k, i in enumerate(np.sort(item_ids))}
        self.dev = dev

    def rows(self, item_ids):
        return np.fromiter((self.index[i] for i in item_ids), dtype=np.int64, count=len(item_ids))

    def batch(self, rows):
        p = self.packed[torch.as_tensor(rows, device=self.dev)]                               # B x 65 x 9
        bits = (p.unsqueeze(-1) >> _SHIFTS.to(self.dev)) & 1                                  # B x 65 x 9 x 8
        return bits.flatten(2)[..., :CANVAS].unsqueeze(1).float()                             # B x 1 x 65 x 65


class UrlData:
    def __init__(self, urls: pd.Series, dev):
        ids, _ = encode_urls(urls)
        self.ids = torch.from_numpy(ids).to(dev)
        self.dev = dev

    def batch(self, rows):
        return self.ids[torch.as_tensor(rows, device=self.dev)].long()


def _batches(n, bs, shuffle, rng=None):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _subsample(df, n, seed):
    return df if n is None or len(df) <= n else df.sample(n, random_state=seed)


def _per_group(df, keys, n, seed):
    """At most n rows of every group (for smoke tests). Keeps every column, unlike groupby.apply in pandas 3."""
    if n is None:
        return df
    idx = [g.sample(min(len(g), n), random_state=seed).index for _, g in df.groupby(keys, observed=True)]
    return df.loc[np.concatenate(idx)]


# ----------------------------------------------------------------------------------------------------------------------
# Image VAE
# ----------------------------------------------------------------------------------------------------------------------
def run_image_vae(latent=32, beta=1.0, seed=0, epochs=40, batch=256, lr=1e-3, patience=5, kl_warmup=0,
                  n_score_samples=5, max_train=None, max_eval=None, tag=""):
    name = f"image_vae_L{latent}_b{beta:g}_s{seed}{tag}"
    seed_everything(seed); dev = device(); rng = np.random.default_rng(seed)
    items = load_items(IMAGE_COLS)
    items = items[items["fits_canvas"]]
    train = _subsample(items[(items["split"] == "train") & (items["label"] == 0)], max_train, seed)
    val_b = _subsample(items[(items["split"] == "validation") & (items["label"] == 0)], max_eval, seed)
    evals = items[items["split"].isin(["validation", "test"])]
    evals = _per_group(evals, ["split", "label"], max_eval, seed)
    data = ImageData(np.concatenate([train["item_id"].to_numpy(), evals["item_id"].to_numpy()]), dev)
    tr_rows, vb_rows, ev_rows = data.rows(train["item_id"].to_numpy()), data.rows(val_b["item_id"].to_numpy()), data.rows(evals["item_id"].to_numpy())

    model = ImageVAE(latent, beta).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    d = CANVAS * CANVAS
    history, best, bad = [], np.inf, 0
    print(f"[{name}] device={dev} train={len(tr_rows):,} val_benign={len(vb_rows):,} eval={len(ev_rows):,}")
    for ep in range(epochs):
        t0 = time.time(); model.train()
        b_eff = beta * min(1.0, (ep + 1) / kl_warmup) if kl_warmup else beta
        sums = np.zeros(2); kl_dims = torch.zeros(latent, device=dev); n = 0
        for r in _batches(len(tr_rows), batch, True, rng):
            x = data.batch(tr_rows[r])
            nll, kl = model(x)
            loss = (nll + b_eff * kl.sum(1)).mean() / d
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            sums += [nll.sum().item(), kl.sum().item()]; kl_dims += kl.detach().sum(0); n += len(r)
        val = _score_vae(model, data, vb_rows, batch, n_samples=1, seed=seed)
        kl_dims = (kl_dims / n).cpu().numpy()
        h = {"epoch": ep, "beta_eff": b_eff, "train_nll_per_module": sums[0] / n / d, "train_kl_per_item": sums[1] / n,
             "active_units": int((kl_dims > 0.01).sum()), "val_benign_neg_elbo": float(val["neg_elbo"].mean()),
             "val_benign_nll": float(val["nll"].mean()), "seconds": time.time() - t0}
        history.append(h)
        print(f"[{name}] ep {ep:2d} nll/mod {h['train_nll_per_module']:.4f} KL {h['train_kl_per_item']:.2f} "
              f"active {h['active_units']}/{latent} val -ELBO/mod {h['val_benign_neg_elbo']:.4f} ({h['seconds']:.0f}s)", flush=True)
        if h["val_benign_neg_elbo"] < best - 1e-5:
            best, bad = h["val_benign_neg_elbo"], 0
            CKPT_DIR.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), CKPT_DIR / f"{name}.pt")
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=dev))
    sc = _score_vae(model, data, ev_rows, batch, n_score_samples, seed)
    out = pd.DataFrame({"item_id": evals["item_id"].to_numpy(), **{k: v for k, v in sc.items()}})
    val_mask = (evals["split"] == "validation").to_numpy()
    val_auc = {k: auroc(out.loc[val_mask, k], evals.loc[val_mask, "label"]) for k in ("nll", "neg_elbo")}
    _save(name, out, {"model": "image_vae", "latent": latent, "beta": beta, "seed": seed, "epochs": epochs, "batch": batch,
                      "lr": lr, "patience": patience, "kl_warmup": kl_warmup, "max_train": max_train, "max_eval": max_eval},
          history, {"val_auroc": val_auc, "best_val_benign_neg_elbo": best})
    return name


@torch.no_grad()
def _score_vae(model, data, rows, batch, n_samples, seed):
    model.eval()
    g = torch.Generator(device=data.dev).manual_seed(seed)
    parts = {"nll": [], "neg_elbo": [], "kl": []}
    for r in _batches(len(rows), 2 * batch, False):
        s = model.score(data.batch(rows[r]), n_samples, g)
        for k in parts:
            parts[k].append(s[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Supervised references
# ----------------------------------------------------------------------------------------------------------------------
def run_reference(kind="url", seed=0, epochs=20, batch=512, lr=1e-3, patience=3, max_train=None, max_eval=None, tag=""):
    name = f"{kind}_ref_s{seed}{tag}"
    seed_everything(seed); dev = device(); rng = np.random.default_rng(seed)
    items = load_items(IMAGE_COLS if kind == "image" else ITEM_COLS)
    if kind == "image":
        items = items[items["fits_canvas"]]
    train = items[items["split"] == "train"]
    train = _per_group(train, "label", max_train, seed)
    evals = items[items["split"].isin(["validation", "test"])]
    evals = _per_group(evals, ["split", "label"], max_eval, seed)
    both = pd.concat([train, evals])
    if kind == "image":
        data = ImageData(both["item_id"].to_numpy(), dev)
        tr_rows, ev_rows = data.rows(train["item_id"].to_numpy()), data.rows(evals["item_id"].to_numpy())
        model = ImageCNN().to(dev)
    else:
        data = UrlData(both["url"], dev)
        tr_rows, ev_rows = np.arange(len(train)), np.arange(len(train), len(both))
        model = CharCNN().to(dev)
    y_tr = torch.tensor(train["label"].to_numpy(), dtype=torch.float32, device=dev)   # aligned with tr_rows
    val_mask = (evals["split"] == "validation").to_numpy()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history, best, bad = [], -np.inf, 0
    print(f"[{name}] device={dev} train={len(tr_rows):,} eval={len(ev_rows):,}")
    for ep in range(epochs):
        t0 = time.time(); model.train(); tot = 0.0
        for r in _batches(len(tr_rows), batch, True, rng):
            loss = F.binary_cross_entropy_with_logits(model(data.batch(tr_rows[r])), y_tr[r])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += loss.item() * len(r)
        s = _logits(model, data, ev_rows[val_mask], batch)
        va = auroc(s, evals.loc[val_mask, "label"])
        history.append({"epoch": ep, "train_bce": tot / len(tr_rows), "val_auroc": va, "seconds": time.time() - t0})
        print(f"[{name}] ep {ep:2d} bce {tot / len(tr_rows):.4f} val AUROC {va:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if va > best + 1e-5:
            best, bad = va, 0
            CKPT_DIR.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), CKPT_DIR / f"{name}.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(torch.load(CKPT_DIR / f"{name}.pt", map_location=dev))
    out = pd.DataFrame({"item_id": evals["item_id"].to_numpy(), "logit": _logits(model, data, ev_rows, batch)})
    _save(name, out, {"model": f"{kind}_ref", "seed": seed, "epochs": epochs, "batch": batch, "lr": lr, "patience": patience,
                      "max_train": max_train, "max_eval": max_eval}, history, {"val_auroc": {"logit": best}})
    return name


@torch.no_grad()
def _logits(model, data, rows, batch):
    model.eval()
    return np.concatenate([model(data.batch(rows[r])).float().cpu().numpy() for r in _batches(len(rows), 2 * batch, False)])


# ----------------------------------------------------------------------------------------------------------------------
def _save(name, scores: pd.DataFrame, config: dict, history: list, summary: dict):
    SCORES_DIR.mkdir(parents=True, exist_ok=True); RUNS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_DIR / f"{name}.parquet", index=False)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps({"name": name, "config": config, "history": history, **summary}, indent=2, default=float))
    print(f"[{name}] saved scores ({len(scores):,} rows), validation AUROC {summary['val_auroc']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["image_vae", "image_ref", "url_ref"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl-warmup", type=int, default=0, help="epochs of linear beta warm-up (image VAE)")
    ap.add_argument("--max-train", type=int, help="subsample training rows (per class for references), for smoke tests")
    ap.add_argument("--max-eval", type=int, help="subsample evaluation rows per split and class, for smoke tests")
    ap.add_argument("--tag", default="", help="suffix for the run name")
    a = ap.parse_args()
    common = {k: v for k, v in dict(seed=a.seed, lr=a.lr, max_train=a.max_train, max_eval=a.max_eval, tag=a.tag,
                                    epochs=a.epochs, batch=a.batch).items() if v is not None}
    if a.model == "image_vae":
        run_image_vae(latent=a.latent, beta=a.beta, kl_warmup=a.kl_warmup, **common)
    else:
        run_reference(kind=a.model.split("_")[0], **common)


if __name__ == "__main__":
    main()
