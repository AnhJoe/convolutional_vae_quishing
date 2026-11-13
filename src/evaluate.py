"""Compare a VAE (several seeds) with its floors and its supervised reference on the test split.

    python -m src.evaluate image --vae image_vae_L32_b1_s0 image_vae_L32_b1_s1 ... --score neg_elbo --ref image_ref_s0
    python -m src.evaluate url   --vae url_vae_..._s0 ...                         --score neg_elbo --ref url_ref_s0

The URL VAE only has to write results/scores/<run>.parquet with item_id and a score column, as src/train.py does.
Tables go to results/tables/<track>_*.csv. The population is the intersection of the item_ids every compared model
scored, so a model that skipped items cannot gain from it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_BOOT, RESULTS_DIR, SCORES_DIR
from .metrics import Evaluator, success_criteria
from .prepare_data import load_items

TABLES_DIR = RESULTS_DIR / "tables"
EVAL_COLS = ["item_id", "label", "split", "domain", "url_len", "scheme", "host_is_ip"]


def load_score(run: str, col: str | None = None) -> pd.Series:
    df = pd.read_parquet(SCORES_DIR / f"{run}.parquet")
    if col is None:
        cols = [c for c in df.columns if c != "item_id"]
        if len(cols) != 1:
            raise ValueError(f"{run} has score columns {cols}; pass one")
        col = cols[0]
    return df.set_index("item_id")[col].rename(run)


def evaluate_track(track: str, vae_runs: list[str] | None = None, vae_col: str | None = None, ref_run: str | None = None,
                   n_boot: int = N_BOOT, save: bool = True) -> dict:
    floors = {k: load_score(f"floor_{k}_{track}") for k in ("A", "B")}
    vaes = {r: load_score(r, vae_col) for r in (vae_runs or [])}
    ref = load_score(ref_run) if ref_run else None

    ids = set(floors["A"].index)
    for s in [floors["B"], *vaes.values()] + ([ref] if ref is not None else []):
        ids &= set(s.index)
    items = load_items(EVAL_COLS)
    df = items[items["item_id"].isin(ids)].sort_values("item_id").reset_index(drop=True)
    al = lambda s: s.reindex(df["item_id"]).to_numpy()

    ev = Evaluator(df, n_boot=n_boot)
    scores = {"floor_A": al(floors["A"]), "floor_B": al(floors["B"]), **{r: al(s) for r, s in vaes.items()}}
    if ref is not None:
        scores["reference"] = al(ref)
    summary = ev.summary(scores)
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(TABLES_DIR / f"{track}_summary.csv")
    if not vaes:                       # floors and reference only, e.g. before the URL VAE exists
        return {"summary": summary, "criteria": None, "n_test": len(ev.test)}

    crit = success_criteria(ev, {r: scores[r] for r in vaes}, scores["floor_A"], scores["floor_B"], scores.get("reference"))
    if save:
        crit["strata_table"].to_csv(TABLES_DIR / f"{track}_strata.csv", index=False)
        (TABLES_DIR / f"{track}_criteria.json").write_text(
            json.dumps({k: v for k, v in crit.items() if k != "strata_table"}, indent=2, default=_jsonable))
    return {"summary": summary, "criteria": crit, "n_test": len(ev.test)}


def _jsonable(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track", choices=["url", "image"])
    ap.add_argument("--vae", nargs="*", default=[], help="run names, one per seed (omit for floors and reference only)")
    ap.add_argument("--score", default=None, help="score column of the VAE runs (nll or neg_elbo)")
    ap.add_argument("--ref", default=None, help="supervised reference run name")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    res = evaluate_track(a.track, a.vae, a.score, a.ref, a.n_boot)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(f"test items: {res['n_test']:,}\n")
    print(res["summary"].round(4).T, "\n")
    if res["criteria"] is None:
        return
    print(res["criteria"]["strata_table"].round(4), "\n")
    print(json.dumps({k: v for k, v in res["criteria"].items() if k != "strata_table"}, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
