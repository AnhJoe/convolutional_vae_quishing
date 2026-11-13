"""Confidence intervals, paired comparisons, strata and seed aggregation on top of src/evaluate.py.

src/evaluate.py turns one score file into point metrics. This file reads the same score files (results/<model>/<score>/
seed<k>/{validation,test}.csv.gz, checked with `load_scores`) and adds what the proposal's success criteria need:

    domain bootstrap 95% CIs    whole domains (the split key) are resampled, never single rows. One draw of domain
                                weights is shared by every model, so differences between models are paired.
    paired AUROC difference     model minus floor on the same test items and the same draws. "Beats" means the 95% CI of
                                the difference is above zero.
    strata                      URL-length bin and scheme for every model, QR version for image models (Micro QR is
                                version 0, its own stratum). A stratum is tested only if it has MIN_PER_CLASS items of
                                each class. IP hosts are reported as a share only.
    seed aggregation            mean and spread over seeds, and the worst seed, which must still beat the floor.
    gap closed                  (model - floor) / (reference - floor) in AUROC. Descriptive, no threshold.

Typical use (3_image_vae.ipynb, 4_url_vae.ipynb):

    cmp = Comparison(RESULTS_DIR, items, SCOPE)
    cmp.summary([("floorA_image", "mahalanobis", 0), ("image_vae", "neg_elbo", 0), ...])
    report = cmp.success_criteria(model="image_vae", score="neg_elbo", seeds=[0, 1, 2, 3, 4],
                                  floor_a=("floorA_image", "mahalanobis"), floor_b=("floorA_image", "mahalanobis"),
                                  reference=("image_supervised", "prob_malicious", 0))

Every metric is computed as a weighted statistic on scores sorted once, so 1,000 draws over the 150k test items take
seconds. The point values (all weights 1) equal those of src/evaluate.py, which `check_against_evaluate` confirms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import FPR_LEVELS, evaluate_run, load_scores, run_dir, threshold_at_fpr

N_BOOT = 1000
MIN_PER_CLASS = 200        # items of each class a stratum needs for a stable AUROC
ALPHA = 0.05


# ----------------------------------------------------------------------------------------------------------------------
# Weighted metrics on scores sorted once
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class _Sorted:
    order: np.ndarray       # argsort of the scores, ascending
    gid: np.ndarray         # tie-group id of each sorted position (equal scores share an id)
    n_groups: int
    y: np.ndarray           # labels in sorted order (True = malicious)

    @classmethod
    def of(cls, score, y) -> "_Sorted":
        s = np.asarray(score, dtype=np.float64)
        order = np.argsort(s, kind="mergesort")
        s = s[order]
        gid = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])]) if len(s) else np.zeros(0, int)
        return cls(order, gid, int(gid[-1]) + 1 if len(s) else 0, np.asarray(y).astype(bool)[order])

    def _per_group(self, w):
        ws = np.ones(len(self.order)) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        return (np.bincount(self.gid, ws * self.y, minlength=self.n_groups),
                np.bincount(self.gid, ws * ~self.y, minlength=self.n_groups))

    def auroc(self, w=None) -> float:
        gp, gn = self._per_group(w)
        den = gp.sum() * gn.sum()
        return float((gp * (np.cumsum(gn) - 0.5 * gn)).sum() / den) if den > 0 else np.nan

    def auprc(self, w=None) -> float:
        """Average precision with ties as one threshold, as in src/evaluate.auprc."""
        gp, gn = self._per_group(w)
        tp, fp = np.cumsum(gp[::-1]), np.cumsum(gn[::-1])
        return float((gp[::-1] * tp / np.maximum(tp + fp, 1e-300)).sum() / gp.sum()) if gp.sum() > 0 else np.nan


def _ci(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan
    return float(np.quantile(v, ALPHA / 2)), float(np.quantile(v, 1 - ALPHA / 2))


class DomainBootstrap:
    """Per-row weights from resampling whole domains with replacement. The same draws serve every model."""

    def __init__(self, domains, n_boot: int = N_BOOT, seed: int = 0):
        codes, uniq = pd.factorize(pd.Series(domains, dtype=object).fillna(""))
        rng = np.random.default_rng(seed)
        n = len(uniq)
        self.codes = codes
        self.draws = [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)]

    def __iter__(self):
        for c in self.draws:
            yield c[self.codes].astype(np.float64)

    def __len__(self):
        return len(self.draws)


# ----------------------------------------------------------------------------------------------------------------------
# Comparison of runs saved with save_and_evaluate
# ----------------------------------------------------------------------------------------------------------------------
class Comparison:
    """All runs of one scope ("all" for URL models, "canvas" for image models) on the same test items and draws."""

    def __init__(self, results_dir: Path | str, items: pd.DataFrame, scope: str, n_boot: int = N_BOOT, seed: int = 0):
        self.results_dir, self.items, self.scope = Path(results_dir), items, scope
        test = items[items["split"] == "test"]
        if scope == "canvas":
            test = test[test["fits_canvas"]]
        self.test = test.sort_values("item_id").reset_index(drop=True)
        self.y = self.test["label"].to_numpy().astype(bool)
        self.boot = DomainBootstrap(self.test["domain"], n_boot, seed)
        self._runs: dict = {}
        self._sorted: dict = {}

    # -- loading -------------------------------------------------------------------------------------------------------
    def run(self, model: str, score: str, seed: int) -> dict:
        """Validation benign scores and test scores (aligned with self.test) of one saved run."""
        key = (model, score, int(seed))
        if key not in self._runs:
            folder = run_dir(self.results_dir, model, score, seed)
            val = load_scores(folder / "validation.csv.gz", self.items, "validation", self.scope)
            test = load_scores(folder / "test.csv.gz", self.items, "test", self.scope).sort_values("item_id")
            self._runs[key] = {"val_benign": val.loc[val["label"] == 0, "score"].to_numpy(float),
                               "test": test["score"].to_numpy(float)}
        return self._runs[key]

    def _srt(self, key, mask=None) -> _Sorted:
        k = (key, None if mask is None else hash(mask.tobytes()))
        if k not in self._sorted:
            s = self.run(*key)["test"]
            self._sorted[k] = _Sorted.of(s, self.y) if mask is None else _Sorted.of(s[mask], self.y[mask])
        return self._sorted[k]

    @staticmethod
    def _name(key) -> str:
        return f"{key[0]}/{key[1]}/seed{key[2]}"

    # -- one table row per run, with CIs -------------------------------------------------------------------------------
    def summary(self, runs: list[tuple[str, str, int]]) -> pd.DataFrame:
        """AUROC, AUPRC and recall at 1% and 5% FPR (thresholds from validation benign scores) with domain bootstrap
        95% CIs, plus the test FPR those thresholds give and the prevalence."""
        rows = []
        for key in runs:
            r, srt = self.run(*key), self._srt(key)
            flags = {f: r["test"] > threshold_at_fpr(r["val_benign"], f) for f in FPR_LEVELS}

            def stats(w=None):
                out = {"auroc": srt.auroc(w), "auprc": srt.auprc(w)}
                ww = np.ones(len(self.y)) if w is None else w
                for f, fl in flags.items():
                    out[f"recall_at_{int(round(f * 100))}pct_fpr"] = float((ww * fl * self.y).sum() / (ww * self.y).sum())
                return out

            point, draws = stats(), pd.DataFrame([stats(w) for w in self.boot])
            row = {"run": self._name(key)}
            for k, v in point.items():
                row[k] = v
                row[f"{k}_lo"], row[f"{k}_hi"] = _ci(draws[k])
            for f, fl in flags.items():
                row[f"test_fpr_at_{int(round(f * 100))}pct"] = float(fl[~self.y].mean())
            row["prevalence"] = float(self.y.mean())
            rows.append(row)
        return pd.DataFrame(rows).set_index("run")

    # -- paired difference ---------------------------------------------------------------------------------------------
    def paired_diff(self, a, b, mask=None) -> dict:
        """AUROC(a) - AUROC(b) on the test items (or a subset), with a paired domain bootstrap 95% CI."""
        sa, sb = self._srt(a, mask), self._srt(b, mask)
        diffs = [sa.auroc(w if mask is None else w[mask]) - sb.auroc(w if mask is None else w[mask]) for w in self.boot]
        lo, hi = _ci(diffs)
        return {"diff": sa.auroc() - sb.auroc(), "lo": lo, "hi": hi, "beats": bool(lo > 0)}

    # -- strata --------------------------------------------------------------------------------------------------------
    def strata(self) -> dict[str, pd.Series]:
        out = {"url_len_bin": self.test["url_len_bin"].astype(str), "scheme": self.test["scheme"].astype(str)}
        if self.scope == "canvas":
            v = self.test["qr_version"]
            out["qr_version"] = np.where(v == 0, "micro", v.astype("Int64").astype(str))
            out["qr_version"] = pd.Series(out["qr_version"], index=self.test.index)
        return out

    def stratified(self, model, floor) -> pd.DataFrame:
        """Per stratum: counts, AUROC of model and floor, and their paired difference when the stratum is large enough."""
        rows = []
        for kind, lab in self.strata().items():
            for value in sorted(pd.unique(lab)):
                m = (lab == value).to_numpy()
                nb, nm = int((m & ~self.y).sum()), int((m & self.y).sum())
                row = {"stratum": kind, "value": value, "n_benign": nb, "n_malicious": nm, "tested": min(nb, nm) >= MIN_PER_CLASS}
                if row["tested"]:
                    row["auroc_model"] = self._srt(model, m).auroc()
                    row["auroc_floor"] = self._srt(floor, m).auroc()
                    d = self.paired_diff(model, floor, m)
                    row.update(diff=d["diff"], diff_lo=d["lo"], diff_hi=d["hi"], beats=d["beats"])
                rows.append(row)
        ip = self.test["host_is_ip"].to_numpy().astype(bool)
        rows.append({"stratum": "ip_host (share only)", "value": "IP host", "n_benign": int((ip & ~self.y).sum()),
                     "n_malicious": int((ip & self.y).sum()), "tested": False})
        return pd.DataFrame(rows)

    # -- gap closed ----------------------------------------------------------------------------------------------------
    def gap_closed(self, model, floor, reference) -> dict:
        s = {k: self._srt(k) for k in (model, floor, reference)}

        def g(w=None):
            fl = s[floor].auroc(w)
            den = s[reference].auroc(w) - fl
            return (s[model].auroc(w) - fl) / den if abs(den) > 1e-12 else np.nan

        lo, hi = _ci([g(w) for w in self.boot])
        return {"gap_closed": g(), "lo": lo, "hi": hi}

    # -- the proposal's success criteria -------------------------------------------------------------------------------
    def success_criteria(self, model: str, score: str, seeds, floor_a: tuple[str, str], floor_b: tuple[str, str],
                         reference: tuple[str, str, int] | None = None) -> dict:
        """Beats floor A (paired CI above zero), holds in a majority of tested strata, the worst seed still beats floor A,
        floor B reported alongside, and the gap closed to the reference. Strata and the gap use the median seed."""
        fa, fb = (*floor_a, 0), (*floor_b, 0)
        keys = [(model, score, int(s)) for s in seeds]
        auc = {k: self._srt(k).auroc() for k in keys}
        ranked = sorted(keys, key=auc.get)
        worst, rep = ranked[0], ranked[len(ranked) // 2]
        strata = self.stratified(rep, fa)
        tested = strata[strata["tested"]]
        n_beats = int(tested["beats"].astype(bool).sum()) if len(tested) else 0
        out = {
            "seeds": {self._name(k): v for k, v in auc.items()},
            "auroc_mean": float(np.mean(list(auc.values()))),
            "auroc_std": float(np.std(list(auc.values()), ddof=1)) if len(auc) > 1 else 0.0,
            "representative_seed": self._name(rep),
            "beats_floor_a": self.paired_diff(rep, fa),
            "vs_floor_b": self.paired_diff(rep, fb),
            "holds_within_strata": {"tested": int(len(tested)), "beats": n_beats, "majority": n_beats > len(tested) / 2},
            "worst_seed_beats_floor_a": {"seed": self._name(worst), **self.paired_diff(worst, fa)},
            "strata": strata,
        }
        if reference is not None:
            out["gap_closed"] = self.gap_closed(rep, fa, tuple(reference))
        return out

    # -- agreement with src/evaluate.py --------------------------------------------------------------------------------
    def check_against_evaluate(self, runs) -> pd.DataFrame:
        """Point metrics here and in evaluate_run must agree. Returns the largest absolute difference per run."""
        rows = []
        for key in runs:
            ref = evaluate_run(run_dir(self.results_dir, *key), self.items, self.scope).iloc[0]
            srt, r = self._srt(key), self.run(*key)
            here = {"auroc": srt.auroc(), "auprc": srt.auprc()}
            for f in FPR_LEVELS:
                t = int(round(f * 100))
                here[f"recall_at_{t}pct_fpr"] = float((r["test"][self.y] > threshold_at_fpr(r["val_benign"], f)).mean())
            rows.append({"run": self._name(key), "max_abs_diff": max(abs(here[k] - ref[k]) for k in here)})
        return pd.DataFrame(rows).set_index("run")


def criteria_table(report: dict) -> pd.DataFrame:
    """The success criteria of one report as a small readable table."""
    b, w, s = report["beats_floor_a"], report["worst_seed_beats_floor_a"], report["holds_within_strata"]
    rows = [
        ("beats floor A (paired 95% CI above 0)", f"{b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}]", b["beats"]),
        ("holds within strata (majority of tested)", f"{s['beats']} of {s['tested']}", s["majority"]),
        ("worst seed beats floor A", f"{w['seed']}: {w['diff']:+.4f} [{w['lo']:+.4f}, {w['hi']:+.4f}]", w["beats"]),
        ("vs floor B (reported)", f"{report['vs_floor_b']['diff']:+.4f} [{report['vs_floor_b']['lo']:+.4f}, {report['vs_floor_b']['hi']:+.4f}]",
         report["vs_floor_b"]["beats"]),
        ("AUROC over seeds", f"{report['auroc_mean']:.4f} +/- {report['auroc_std']:.4f}", None),
    ]
    if "gap_closed" in report:
        g = report["gap_closed"]
        rows.append(("gap closed to the reference (descriptive)", f"{g['gap_closed']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}]", None))
    return pd.DataFrame(rows, columns=["criterion", "value", "met"]).set_index("criterion")
