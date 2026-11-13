"""Evaluation shared by every model: AUROC, AUPRC, recall at a fixed FPR, domain bootstrap CIs, paired comparisons, strata.

Conventions (from the proposal):
- Higher score = more likely malicious.
- FPR thresholds are set on validation benign scores only, then applied to test.
- Confidence intervals resample whole domains (the split key), not single rows.
- Every metric is computed as a weighted statistic, so one bootstrap replicate is one vector of domain weights and the
  scores are sorted only once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FPR_TARGETS, MIN_PER_CLASS_STRATUM, N_BOOT, URL_LEN_BIN_EDGES


# ----------------------------------------------------------------------------------------------------------------------
# Weighted metrics on a pre-sorted score vector
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class _Sorted:
    order: np.ndarray      # argsort of the scores, ascending
    gid: np.ndarray        # tie-group id of each sorted position (equal scores share an id)
    n_groups: int
    y: np.ndarray          # labels in sorted order

    @classmethod
    def of(cls, score: np.ndarray, y: np.ndarray) -> "_Sorted":
        score = np.asarray(score, dtype=np.float64)
        if np.isnan(score).any():
            raise ValueError("scores contain NaN")
        order = np.argsort(score, kind="mergesort")
        s = score[order]
        gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])
        return cls(order, gid, int(gid[-1]) + 1, np.asarray(y)[order].astype(bool))

    def _class_weights(self, w):
        ws = np.ones(len(self.order)) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        gp = np.bincount(self.gid, ws * self.y, minlength=self.n_groups)
        gn = np.bincount(self.gid, ws * ~self.y, minlength=self.n_groups)
        return gp, gn

    def auroc(self, w=None) -> float:
        gp, gn = self._class_weights(w)
        neg_below = np.cumsum(gn) - gn
        return float((gp * (neg_below + 0.5 * gn)).sum() / (gp.sum() * gn.sum()))

    def auprc(self, w=None) -> float:
        """Average precision, ties treated as one threshold (the scikit-learn convention)."""
        gp, gn = self._class_weights(w)
        tp = np.cumsum(gp[::-1]); fp = np.cumsum(gn[::-1])          # counts at or above each threshold, high to low
        prec = tp / np.maximum(tp + fp, 1e-300)
        return float((gp[::-1] * prec).sum() / gp.sum())


def auroc(score, y, w=None) -> float:
    return _Sorted.of(score, y).auroc(w)


def auprc(score, y, w=None) -> float:
    return _Sorted.of(score, y).auprc(w)


def fpr_threshold(benign_val_scores: np.ndarray, fpr: float) -> float:
    """Score above which `fpr` of validation benign items fall (items with score > threshold are flagged)."""
    return float(np.quantile(np.asarray(benign_val_scores, dtype=np.float64), 1 - fpr, method="higher"))


def _rate(flag, mask, w):
    w = np.ones(len(flag)) if w is None else w
    return float((w * flag * mask).sum() / (w * mask).sum())


# ----------------------------------------------------------------------------------------------------------------------
# Domain bootstrap
# ----------------------------------------------------------------------------------------------------------------------
class DomainBootstrap:
    """Draws per-row weights by resampling whole domains with replacement. The same draws are reused for every model,
    which makes model differences paired."""

    def __init__(self, domains, n_boot: int = N_BOOT, seed: int = 0):
        self.codes, uniq = pd.factorize(pd.Series(domains).to_numpy())
        rng = np.random.default_rng(seed)
        n = len(uniq)
        self.counts = [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]

    def __iter__(self):
        for c in self.counts:
            yield c[self.codes].astype(np.float64)

    def __len__(self):
        return len(self.counts)


def _ci(values, alpha=0.05):
    v = np.asarray(values)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation frame
# ----------------------------------------------------------------------------------------------------------------------
def strata_of(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Stratum labels per row: URL-length bin and URL scheme (the proposal's strata)."""
    edges = list(URL_LEN_BIN_EDGES)
    labels = [f"{a}-{b - 1}" if b < 10**9 else f"{a}+" for a, b in zip(edges[:-1], edges[1:])]
    return {"url_len_bin": pd.cut(df["url_len"], edges, right=False, labels=labels).astype(str),
            "scheme": df["scheme"].astype(str)}


class Evaluator:
    """Holds the validation and test rows of one evaluation population (e.g. all items for URL models, the items that fit
    the canvas for image models) and the domain bootstrap over its test rows.

    `scores` given to the methods are dicts {model name: array aligned with the rows of `df`}.
    """

    def __init__(self, df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 0):
        df = df.reset_index(drop=True)
        self.val_benign = ((df["split"] == "validation") & (df["label"] == 0)).to_numpy()
        self.test_mask = (df["split"] == "test").to_numpy()
        self.test = df[self.test_mask].reset_index(drop=True)
        self.y = self.test["label"].to_numpy().astype(bool)
        self.boot = DomainBootstrap(self.test["domain"], n_boot, seed)
        self.strata = strata_of(self.test)
        self._sorted_cache: dict = {}

    def _sorted(self, name, s, mask=None):
        key = (name, None if mask is None else hash(mask.tobytes()))
        if key not in self._sorted_cache:
            self._sorted_cache[key] = _Sorted.of(s if mask is None else s[mask], self.y if mask is None else self.y[mask])
        return self._sorted_cache[key]

    def _split(self, s):
        s = np.asarray(s, dtype=np.float64)
        return s[self.val_benign], s[self.test_mask]

    # -- headline table ------------------------------------------------------------------------------------------------
    def summary(self, scores: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        for name, s in scores.items():
            val_b, st = self._split(s)
            srt = self._sorted(name, st)
            thr = {f: fpr_threshold(val_b, f) for f in FPR_TARGETS}
            flags = {f: st > t for f, t in thr.items()}

            def stats(w=None):
                out = {"AUROC": srt.auroc(w), "AUPRC": srt.auprc(w)}
                for f in FPR_TARGETS:
                    out[f"recall@{f:.0%}FPR"] = _rate(flags[f], self.y, w)
                return out

            point = stats()
            boots = pd.DataFrame([stats(w) for w in self.boot])
            row = {"model": name}
            for k, v in point.items():
                lo, hi = _ci(boots[k])
                row[k] = v; row[f"{k} lo"] = lo; row[f"{k} hi"] = hi
            for f in FPR_TARGETS:
                row[f"test FPR @{f:.0%}"] = _rate(flags[f], ~self.y, None)
            row["prevalence"] = self.y.mean()
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    # -- paired comparison ---------------------------------------------------------------------------------------------
    def paired_auroc_diff(self, a_name, a, b_name, b, mask=None) -> dict:
        """AUROC(a) - AUROC(b) on test rows (optionally a subset), with a paired domain bootstrap CI."""
        _, sa = self._split(a); _, sb = self._split(b)
        ra, rb = self._sorted(a_name, sa, mask), self._sorted(b_name, sb, mask)
        diffs = [ra.auroc(w if mask is None else w[mask]) - rb.auroc(w if mask is None else w[mask]) for w in self.boot]
        lo, hi = _ci(diffs)
        return {"diff": ra.auroc() - rb.auroc(), "lo": lo, "hi": hi, "beats": lo > 0}

    def stratified(self, a_name, a, b_name, b) -> pd.DataFrame:
        """Per stratum: item counts, AUROC of both models and the paired difference. Strata with fewer than
        MIN_PER_CLASS_STRATUM items of either class are listed with counts only."""
        _, sa = self._split(a); _, sb = self._split(b)
        rows = []
        for kind, lab in self.strata.items():
            for value in sorted(lab.unique()):
                m = (lab == value).to_numpy()
                n_b, n_m = int((m & ~self.y).sum()), int((m & self.y).sum())
                row = {"stratum": kind, "value": value, "n benign": n_b, "n malicious": n_m}
                if min(n_b, n_m) >= MIN_PER_CLASS_STRATUM:
                    row[f"AUROC {a_name}"] = self._sorted(a_name, sa, m).auroc()
                    row[f"AUROC {b_name}"] = self._sorted(b_name, sb, m).auroc()
                    d = self.paired_auroc_diff(a_name, a, b_name, b, m)
                    row.update({"diff": d["diff"], "diff lo": d["lo"], "diff hi": d["hi"], "beats": d["beats"]})
                rows.append(row)
        ip = self.test["host_is_ip"].to_numpy()
        rows.append({"stratum": "ip_host (share only)", "value": "IP host", "n benign": int((ip & ~self.y).sum()),
                     "n malicious": int((ip & self.y).sum())})
        return pd.DataFrame(rows)

    def gap_closed(self, model_name, model, floor_name, floor, ref_name, ref) -> dict:
        """(model - floor) / (reference - floor) in AUROC, with a paired bootstrap CI. Descriptive, no threshold."""
        s = {n: self._split(v)[1] for n, v in [(model_name, model), (floor_name, floor), (ref_name, ref)]}
        r = {n: self._sorted(n, v) for n, v in s.items()}

        def g(w=None):
            fl = r[floor_name].auroc(w)
            den = r[ref_name].auroc(w) - fl
            return (r[model_name].auroc(w) - fl) / den if abs(den) > 1e-12 else np.nan

        gs = np.array([g(w) for w in self.boot])
        lo, hi = _ci(gs[np.isfinite(gs)]) if np.isfinite(gs).any() else (np.nan, np.nan)
        return {"gap_closed": g(), "lo": lo, "hi": hi}


def success_criteria(ev: Evaluator, vae_runs: dict[str, np.ndarray], floor_a: np.ndarray, floor_b: np.ndarray,
                     reference: np.ndarray | None = None) -> dict:
    """The proposal's success criteria for one VAE configuration run with several seeds.

    vae_runs: {run name (one per seed): scores}. The worst seed by test AUROC is the one checked for seed robustness.
    """
    per_seed = {n: ev.paired_auroc_diff(n, s, "floor_A", floor_a) for n, s in vae_runs.items()}
    auc = {n: ev._sorted(n, ev._split(s)[1]).auroc() for n, s in vae_runs.items()}
    worst = min(auc, key=auc.get)
    rep = sorted(auc, key=auc.get)[len(auc) // 2]   # median seed is the representative run for strata and gap
    strata = ev.stratified(rep, vae_runs[rep], "floor_A", floor_a)
    tested = strata.dropna(subset=["beats"]) if "beats" in strata else strata.iloc[0:0]
    out = {
        "seed_auroc": auc,
        "seed_auroc_mean": float(np.mean(list(auc.values()))), "seed_auroc_std": float(np.std(list(auc.values()), ddof=1)) if len(auc) > 1 else 0.0,
        "beats_floor_A": per_seed[rep],
        "vs_floor_B": ev.paired_auroc_diff(rep, vae_runs[rep], "floor_B", floor_b),
        "holds_within_strata": {"n_strata_tested": int(len(tested)), "n_beats": int(tested["beats"].astype(bool).sum()) if len(tested) else 0},
        "seed_robust": {"worst_seed": worst, **per_seed[worst]},
        "representative_seed": rep,
        "strata_table": strata,
    }
    out["holds_within_strata"]["majority"] = out["holds_within_strata"]["n_beats"] > out["holds_within_strata"]["n_strata_tested"] / 2
    if reference is not None:
        out["gap_closed"] = ev.gap_closed(rep, vae_runs[rep], "floor_A", floor_a, "reference", reference)
    return out
