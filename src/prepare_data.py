"""Build the items table and the QR grid cache that every model and the evaluation read.

    python -m src.prepare_data                 # full run (about 10 minutes for the images on 16 processes)
    python -m src.prepare_data --skip-images   # URL side only (floors, URL models)

Outputs in data/processed/:
- items.parquet    one row per kept QR code: item_id, label, split, url, URL features, domain, image stats
- qr_grids.npy     (N, CANVAS, ceil(CANVAS/8)) uint8, each code as one bit per module, anchored top-left on the canvas,
                   row i belongs to item_id i. Codes larger than the canvas are left blank and have fits_canvas = False.
- prepare_summary.json  row counts per split and class, drops, exclusions

The cleaning and split rules are the ones decided in notebooks/1_eda.ipynb (sections 2, 6, 7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CANVAS, CLASS_FOLDERS, MODULE_PX, PROC_DIR, URL_MAX_LEN, find_raw_dir
from .features import norm_url, url_features
from .splits import DEFAULT_FRACS, DEFAULT_SEED, SPLIT_NAMES, grouped_split

ROW_BYTES = (CANVAS + 7) // 8


# ----------------------------------------------------------------------------------------------------------------------
# URL table
# ----------------------------------------------------------------------------------------------------------------------
def _read_class_csv(folder: Path) -> pd.DataFrame:
    paths = sorted(folder.glob("all_generated_urls_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no all_generated_urls_*.csv in {folder}")
    digests = {hashlib.md5(p.read_bytes()).hexdigest() for p in paths}
    if len(digests) > 1:
        raise ValueError(f"{folder} holds several different URL CSVs: {[p.name for p in paths]}")
    return pd.read_csv(paths[0], keep_default_na=False)


def build_items(raw_dir: Path, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, dict]:
    frames = []
    for cls, folder in CLASS_FOLDERS.items():
        f = _read_class_csv(raw_dir / folder)
        f["cls"] = cls
        frames.append(f)
    raw = pd.concat(frames, ignore_index=True)
    raw["label"] = (raw["cls"] == "malicious").astype("int8")
    raw["file"] = raw["qr_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1]
    raw["url_norm"] = norm_url(raw["url"])
    feat = pd.concat([raw[["cls", "label", "index", "file", "url", "url_norm"]], url_features(raw)], axis=1)

    # Removal rules (EDA section 7): empty URLs, extra copies inside a class, every copy of a URL found in both classes
    shared = set(feat.loc[feat["cls"] == "benign", "url"]) & set(feat.loc[feat["cls"] == "malicious", "url"])
    reason = pd.Series("", index=feat.index)
    reason[feat.duplicated(["cls", "url"], keep="first")] = "duplicate_in_class"
    reason[feat["url"].isin(shared)] = "in_both_classes"
    reason[feat["url"].str.strip().eq("")] = "empty_url"
    drops = reason[reason != ""].value_counts().to_dict()
    items = feat[reason == ""].reset_index(drop=True)

    # Split rule (EDA section 6): each class grouped by domain, 70/15/15, largest domains to train
    split = np.empty(len(items), dtype=np.int8)
    for lab in (0, 1):
        m = (items["label"] == lab).to_numpy()
        split[m] = grouped_split(items.loc[m, "domain"], DEFAULT_FRACS, seed=seed)
    items["split"] = pd.Categorical.from_codes(split, SPLIT_NAMES)
    items.insert(0, "item_id", np.arange(len(items), dtype=np.int32))
    items["truncated"] = items["url_len"] > URL_MAX_LEN
    return items, {"dropped": {k: int(v) for k, v in drops.items()}}


# ----------------------------------------------------------------------------------------------------------------------
# QR grids
# ----------------------------------------------------------------------------------------------------------------------
def _grid_of(path: str):
    """Read one PNG and return (packed canvas, modules, border, dark_frac, fits, error)."""
    from PIL import Image

    try:
        a = np.array(Image.open(path))
        dark = ~a if a.dtype == bool else (a < 128)
        if dark.ndim != 2 or dark.shape[0] != dark.shape[1] or dark.shape[0] % MODULE_PX:
            raise ValueError(f"unexpected shape {dark.shape}")
        g = dark[MODULE_PX // 2::MODULE_PX, MODULE_PX // 2::MODULE_PX]
        side = g.shape[0]
        cols = np.flatnonzero(g.any(axis=0))
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        canvas = np.zeros((CANVAS, CANVAS), dtype=bool)
        fits = side <= CANVAS
        if fits:
            canvas[:side, :side] = g
        return np.packbits(canvas, axis=1), x1 - x0, x0, float(g.mean()), fits, ""
    except Exception as e:  # keep going, record the failure
        return np.zeros((CANVAS, ROW_BYTES), np.uint8), -1, -1, np.nan, False, f"{type(e).__name__}: {e}"


def build_grids(items: pd.DataFrame, raw_dir: Path, out_path: Path, workers: int = 16) -> pd.DataFrame:
    folder = items["cls"].map({c: str(raw_dir / f / "qrs") for c, f in CLASS_FOLDERS.items()})
    paths = (folder + "/" + items["file"]).tolist()
    grids = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8, shape=(len(items), CANVAS, ROW_BYTES))
    stats = np.zeros((len(items), 3)); fits = np.zeros(len(items), bool); errors = [""] * len(items)
    with Pool(workers) as pool:
        for i, (g, mod, border, dark, ok, err) in enumerate(pool.imap(_grid_of, paths, chunksize=512)):
            grids[i] = g; stats[i] = (mod, border, dark); fits[i] = ok; errors[i] = err
            if i % 100_000 == 0:
                print(f"  grids {i:,}/{len(paths):,}", flush=True)
    grids.flush()
    return pd.DataFrame({"modules": stats[:, 0].astype(np.int16), "border": stats[:, 1].astype(np.int16),
                         "dark_frac": stats[:, 2], "fits_canvas": fits, "image_error": errors}, index=items.index)


def load_items(columns=None) -> pd.DataFrame:
    return pd.read_parquet(PROC_DIR / "items.parquet", columns=columns)


def load_grids(mmap: bool = True) -> np.ndarray:
    return np.load(PROC_DIR / "qr_grids.npy", mmap_mode="r" if mmap else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    raw_dir = find_raw_dir()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    print("raw data:", raw_dir)
    items, summary = build_items(raw_dir)
    print(f"items: {len(items):,} kept, dropped {summary['dropped']}")

    if not args.skip_images:
        img = build_grids(items, raw_dir, PROC_DIR / "qr_grids.npy", args.workers)
        items = pd.concat([items, img], axis=1)
        summary["image_errors"] = int((img["image_error"] != "").sum())
        summary["excluded_from_image_models"] = (
            items.loc[~items["fits_canvas"]].groupby("cls").size().reindex(list(CLASS_FOLDERS), fill_value=0).astype(int).to_dict())

    items.to_parquet(PROC_DIR / "items.parquet", index=False)
    counts = items.groupby(["split", "cls"], observed=True).size().unstack()
    summary["counts"] = {s: {c: int(v) for c, v in r.items()} for s, r in counts.iterrows()}
    summary["malicious_share"] = (counts["malicious"] / counts.sum(axis=1)).round(4).to_dict()
    summary["ip_host_share_of_malicious"] = items[items["label"] == 1].groupby("split", observed=True)["host_is_ip"].mean().round(4).to_dict()
    summary["truncated_share"] = items.groupby("cls")["truncated"].mean().round(5).to_dict()
    (PROC_DIR / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
