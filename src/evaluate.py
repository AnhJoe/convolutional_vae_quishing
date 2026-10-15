"""Turn score files into metrics. Every number reported in the project comes from this file.

A model never computes a metric itself. It scores items and saves the scores with `save_scores`. This module reads the
score files, checks them against the items table (src/prepare_data.py), and computes the metrics the same way for every
model, floor and supervised reference.

Score file schema
-----------------
One file holds the scores of one score type, one seed and one split. It is a gzipped CSV with exactly two columns:

    item_id   int     the `item_id` of the items table (data/processed/items.csv.gz). Each id appears once.
    score     float   anomaly score. HIGHER MEANS MORE LIKELY MALICIOUS. No NaN or infinite values.

The rows must be exactly the items the model is scored on, no more and no fewer:

    scope "all"     every item of the split. Use for URL models, URL floors and URL supervised references.
    scope "canvas"  every item of the split with fits_canvas true. Use for image models, image floors and image
                    supervised references (codes larger than the 65 x 65 canvas are excluded, not cropped).

A score whose natural direction is the other way (for example the probability of benign) must be negated before saving.
Scores from a supervised reference are saved the same way (for example the predicted probability of malicious).

Results layout
--------------
    results/<model>/<score>/seed<k>/validation.csv.gz
    results/<model>/<score>/seed<k>/test.csv.gz

    <model>   what produced the score, for example floorA_image, floorB_url, image_vae, url_vae,
              image_supervised, url_supervised. Lower case letters, digits and underscores.
    <score>   which score of that model, for example recon_error, neg_elbo, kl, mahalanobis (image VAE),
              nll_per_char, neg_elbo_per_char (URL VAE), prob_malicious (supervised). A floor has one score, called
              mahalanobis (a floor with several features). The URL floor A has the single feature url_len and is
              scored as an absolute z-score, so its score name is abs_z.
    seed      integer. Models with no randomness (the floors) use seed0.

The validation file is always needed, because the alarm threshold is set on its benign rows. The test file is read only
once the configuration is fixed on validation. `results/` is git-ignored (gz files are), so share score files directly.

Metrics (all on one seed and one split, malicious is the positive class)
------------------------------------------------------------------------
    auroc                 probability that a random malicious item scores above a random benign item (ties count half).
    auprc                 average precision. A random scorer gets the malicious share of the split, about 0.57.
    prevalence            malicious share of the split, the AUPRC of a random scorer.
    recall_at_1pct_fpr    share of malicious items flagged when the threshold lets 1% of VALIDATION benign items through.
    recall_at_5pct_fpr    the same at 5%.
    test_fpr_at_*         share of TEST benign items flagged by that threshold. It should be close to 1% and 5%. A large
                          gap means the validation and test benign score distributions differ.

Threshold rule: flag an item if its score is above the (1 - FPR) quantile of the validation benign scores, so the
threshold never uses test data or malicious items.

Typical use in a model notebook (3_image_vae.ipynb or 4_url_vae.ipynb):

    metrics = save_and_evaluate(RESULTS_DIR, items, SCOPE, "image_vae", "neg_elbo", seed=0,
                                validation=(val_ids, val_scores), test=(test_ids, test_scores))

Command line (from the project root):

    python -m src.evaluate results/image_vae/neg_elbo/seed0 --scope canvas
    python -m src.evaluate results/url_vae/nll_per_char/seed0 --scope all

Not built yet: bootstrap confidence intervals, strata tables, paired comparison with a floor, seed aggregation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.prepare_data import load_items

SCORED_SPLITS = ["validation", "test"]
SCOPES = ["all", "canvas"]
FPR_LEVELS = (0.01, 0.05)


# ----------------------------------------------------------------------------------------------------------------------
# Writing and reading score files
# ----------------------------------------------------------------------------------------------------------------------
def run_dir(results_dir: Path | str, model: str, score: str, seed: int) -> Path:
    """Folder of one model, score type and seed: results/<model>/<score>/seed<k>."""
    return Path(results_dir) / model / score / f"seed{int(seed)}"


def save_scores(results_dir: Path | str, model: str, score: str, seed: int, split: str,
                item_id, values) -> Path:
    """Save the scores of one split in the format described at the top of this file. Returns the file path.

    Only the basic shape is checked here. The match with the items table is checked by `load_scores`.
    """
    if split not in SCORED_SPLITS:
        raise ValueError(f"split must be one of {SCORED_SPLITS}, got {split!r}")
    t = pd.DataFrame({"item_id": np.asarray(item_id, dtype="int64"), "score": np.asarray(values, dtype="float64")})
    if len(t) == 0 or not np.isfinite(t["score"]).all() or t["item_id"].duplicated().any():
        raise ValueError("scores must be non-empty, finite, and have one row per item_id")
    folder = run_dir(results_dir, model, score, seed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{split}.csv.gz"
    t.to_csv(path, index=False)
    return path


def load_scores(path: Path | str, items: pd.DataFrame, split: str, scope: str) -> pd.DataFrame:
    """Read one score file, check it against the items table, and return the items of that split with their score.

    The result has the items table columns plus `score`, so strata and domains are available to later steps.
    Raises ValueError with the reason if the file does not follow the schema.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    t = pd.read_csv(path)
    if list(t.columns) != ["item_id", "score"]:
        raise ValueError(f"{path}: columns must be exactly item_id, score. Found {list(t.columns)}")
    if t["item_id"].duplicated().any():
        raise ValueError(f"{path}: item_id appears more than once")
    if not np.isfinite(t["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: scores contain NaN or infinite values")

    expected = items[items["split"] == split]
    if scope == "canvas":
        expected = expected[expected["fits_canvas"]]
    missing = np.setdiff1d(expected["item_id"].to_numpy(), t["item_id"].to_numpy())
    extra = np.setdiff1d(t["item_id"].to_numpy(), expected["item_id"].to_numpy())
    if len(missing) or len(extra):
        raise ValueError(f"{path}: does not match the {split} items for scope {scope!r} "
                         f"({len(missing):,} missing, {len(extra):,} not expected)")
    return expected.merge(t, on="item_id", how="left", validate="one_to_one").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------
def auroc(y, score) -> float:
    """Area under the ROC curve from ranks (Mann-Whitney). y is 1 for malicious. Ties count half."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    _, start, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(start + (counts + 1) / 2, counts)      # average 1-based rank within each group of ties
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, score) -> float:
    """Average precision: precision at each distinct threshold, weighted by the gain in recall. y is 1 for malicious."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[s[1:] != s[:-1], True]                             # one cut per distinct score
    tp = np.cumsum(y)[last]
    fp = np.cumsum(~y)[last]
    precision, recall = tp / (tp + fp), tp / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def threshold_at_fpr(benign_validation_scores, fpr: float) -> float:
    """Score above which an item is flagged, so at most `fpr` of the validation benign items are flagged."""
    return float(np.quantile(np.asarray(benign_validation_scores, dtype=float), 1 - fpr, method="higher"))


def flagged_share(scores, threshold: float) -> float:
    """Share of the given scores strictly above the threshold."""
    s = np.asarray(scores, dtype=float)
    return float((s > threshold).mean()) if len(s) else float("nan")


# ----------------------------------------------------------------------------------------------------------------------
# One run (one model, score type and seed)
# ----------------------------------------------------------------------------------------------------------------------
def evaluate_run(folder: Path | str, items: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Metrics on the test split for one run folder, with thresholds from the validation split. One row."""
    folder = Path(folder)
    val = load_scores(folder / "validation.csv.gz", items, "validation", scope)
    test = load_scores(folder / "test.csv.gz", items, "test", scope)

    row = {"model": folder.parts[-3], "score": folder.parts[-2], "seed": int(folder.name.removeprefix("seed")), "scope": scope,
           "n_benign": int((test["label"] == 0).sum()), "n_malicious": int((test["label"] == 1).sum())}
    row["prevalence"] = row["n_malicious"] / len(test)
    row["auroc"] = auroc(test["label"], test["score"])
    row["auprc"] = auprc(test["label"], test["score"])
    row["validation_auroc"] = auroc(val["label"], val["score"])
    benign_val = val.loc[val["label"] == 0, "score"]
    for fpr in FPR_LEVELS:
        tag = f"{int(round(fpr * 100))}pct"
        thr = threshold_at_fpr(benign_val, fpr)
        row[f"recall_at_{tag}_fpr"] = flagged_share(test.loc[test["label"] == 1, "score"], thr)
        row[f"test_fpr_at_{tag}"] = flagged_share(test.loc[test["label"] == 0, "score"], thr)
    return pd.DataFrame([row])


def save_and_evaluate(results_dir: Path | str, items: pd.DataFrame, scope: str, model: str, score: str, seed: int,
                      validation, test) -> pd.DataFrame:
    """Save the validation and test scores of one run, then return its metrics (one row, see evaluate_run).

    `validation` and `test` are (item_id, scores) pairs, one score per item the model is scored on (see the scope rules
    at the top of this file). The same call works for every model, floor and supervised reference.
    """
    for split, (ids, values) in zip(SCORED_SPLITS, (validation, test)):
        save_scores(results_dir, model, score, seed, split, ids, values)
    return evaluate_run(run_dir(results_dir, model, score, seed), items, scope)


def main() -> None:
    p = argparse.ArgumentParser(description="Metrics for one run folder (see the top of this file for the schema).")
    p.add_argument("run_dir", type=Path, help="results/<model>/<score>/seed<k>")
    p.add_argument("--scope", choices=SCOPES, required=True, help="all: URL models. canvas: image models")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project root (the folder that contains src and data)")
    a = p.parse_args()
    items = load_items(a.root / "data" / "processed")
    out = evaluate_run(a.run_dir, items, a.scope)
    print(out.T.to_string(header=False))


if __name__ == "__main__":
    main()
