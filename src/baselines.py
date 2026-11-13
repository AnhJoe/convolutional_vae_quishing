"""Fit the floors on benign training items and write their validation and test scores.

    python -m src.baselines

Writes results/scores/floor_{A,B}_{url,image}.parquet (item_id, score). The URL floors score every item. The image
floors are fitted and scored on items whose code fits the canvas, the same population as the image models. Image
floor B equals floor A (EDA section 7) and is written anyway so every track has the same files.
"""
from __future__ import annotations

import json

import pandas as pd

from .config import FLOORS, RESULTS_DIR, SCORES_DIR
from .floors import GaussianFloor
from .prepare_data import load_items


def run_floors() -> dict:
    items = load_items()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    pops = {"url": items}
    if "fits_canvas" in items:
        pops["image"] = items[items["fits_canvas"]]
    report = {}
    for floor, spec in FLOORS.items():
        for kind, df in pops.items():
            fl = GaussianFloor(spec[kind]).fit(df[(df["split"] == "train") & (df["label"] == 0)])
            ev = df[df["split"].isin(["validation", "test"])]
            name = f"floor_{floor}_{kind}"
            pd.DataFrame({"item_id": ev["item_id"].to_numpy(), "score": fl.score(ev)}).to_parquet(SCORES_DIR / f"{name}.parquet", index=False)
            report[name] = {"features": spec[kind], "mean": fl.mean_.tolist(), "std": fl.scale_.tolist()}
            print(f"{name}: {spec[kind]} -> {len(ev):,} scores")
    (RESULTS_DIR / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runs" / "floors.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_floors()
