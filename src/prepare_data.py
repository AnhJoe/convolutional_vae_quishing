"""Build the shared items table that every model and the evaluation framework read.

One row is one QR code with its URL (an "item"). Every split, stratum and floor feature lives in this one table, so
all team members train, tune and score on identical data and the evaluation framework can group results without
touching raw files.

notebooks/2_data_processing.ipynb runs these steps in order and checks the result. The same steps can be run from the
project root:

    python -m src.canvas_exclusion_count        # once, measures every kept PNG header (needed for fits_canvas)
    python -m src.prepare_data images           # once, reads every PNG (modules, border, dark fraction). Can be resumed
    python -m src.prepare_data items            # builds data/processed/items.csv.gz and items_summary.json

Run `items` again after `images` finishes so the image columns are filled. `items` alone is enough for URL work.
The model notebooks (3_image_vae.ipynb, 4_url_vae.ipynb) read the table with `load_items` (below), never with a bare
`pd.read_csv`.

Inputs (all written by 1_eda.ipynb or the canvas tool, all under data/processed/):
    manifest.csv.gz           one row per URL with the cleaning decision (`keep`) and the URL features
    image_grid_sizes.csv.gz   grid size and canvas fit for every kept image (src/canvas_exclusion_count.py)
    image_features.csv.gz     modules, border and dark fraction for every image (the `images` stage of this file)

Output: data/processed/items.csv.gz, one row per KEPT row (1,005,700 rows), in manifest order (all benign rows, then all
malicious rows). `item_id` is the row number in that order and is the key that score files use (see src/evaluate.py).

Table schema
------------
Identity and label
    item_id        int      0 .. n-1, position in the table. Stable as long as the manifest and split rule do not change.
    cls            str      "benign" or "malicious", taken from the folder that holds the image (never from the file name).
    label          int      0 for benign, 1 for malicious. Malicious is the positive (anomalous) class.
    index          int      row number in the source CSV of that class.
    file           str      PNG file name, found at data/raw/<class folder>/qrs/<file>.
    url            str      the URL exactly as encoded in the QR code (no cleaning, may lack a scheme).

Split (rule in src/splits.py, decided in 1_eda.ipynb section 6)
    split          str      "train", "validation" or "test". Benign and malicious rows are each split 70/15/15,
                            grouped by domain (all rows of one domain share a split), seed 0.
    domain         str      approximate registered domain, the grouping key. For an IP host it is the last two numbers
                            of the address (for example "115.240"). Empty for 15 rows. Bootstrap CIs resample this key.
    host           str      host name or address of the URL.

Canvas (decided in 1_eda.ipynb section 4: 65 x 65 modules, quiet zone included, 6 pixels per module)
    fits_canvas    bool     True if the full grid (modules plus quiet zone) is at most 65 x 65. Image models use only
                            rows with fits_canvas True and are scored only on those rows. URL models use every row.
    grid           float    full grid size in modules (PNG width divided by 6). Empty for an unreadable image.

URL features (floor B inputs and strata)
    url_len        int      number of characters in the URL.
    url_len_bin    str      "1-32", "33-64", "65-128", "129-256" or ">256". Stratum for URL results. The last bin is
                            exactly the truncated URLs.
    truncated      bool     True if url_len is above 256, the maximum length of the URL model (cut off at 256).
    scheme         str      "http", "https", "other" or "none" (no scheme). Stratum for URL results.
    has_scheme     bool     the URL starts with a scheme.
    is_https       bool     the scheme is https.
    host_is_ip     bool     the host is an IP address. Almost all such rows are malicious, so this group is reported as a
                            share only (the benign side is too small for a stratified AUROC).
    n_host_labels  int      number of dot-separated labels in the host.
    path_len       int      length of the path part.
    digit_ratio    float    digits divided by url_len.

Image features (filled by the `images` stage, empty until it has run)
    side           float    image width in pixels (images are square).
    modules        float    QR modules per side, quiet zone excluded.
    border         float    quiet zone width in modules.
    dark_frac      float    fraction of dark pixels in the whole image (modules and quiet zone).
    qr_version     float    1 to 40 for a standard QR code, 0 for a Micro QR code (fewer than 21 modules per side).
                            Stratum for image results, Micro QR is its own stratum. Empty if the image was not read.

Floor inputs: floor A (image) uses modules and dark_frac, floor A (URL) uses url_len, floor B (URL) adds has_scheme,
host_is_ip, n_host_labels, path_len and digit_ratio. All are columns of this table, fitted on benign train rows only.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.canvas_exclusion_count import CLASS_FOLDER, DEFAULT_CANVAS, DEFAULT_MODULE_PX
from src.splits import DEFAULT_FRACS, DEFAULT_SEED, SPLIT_NAMES, grouped_split

ITEMS_FILE = "items.csv.gz"
IMAGE_FEATURES_FILE = "image_features.csv.gz"
IMAGE_FEATURES_PARTIAL = "image_features_partial.csv"
MAX_URL_LEN = 256                                             # decided in 1_eda.ipynb section 7
URL_LEN_EDGES = [0, 32, 64, 128, MAX_URL_LEN, np.inf]
URL_LEN_LABELS = ["1-32", "33-64", "65-128", "129-256", ">256"]
IMAGE_COLUMNS = ["side", "modules", "border", "dark_frac", "qr_version"]
STRING_COLUMNS = ["cls", "file", "url", "split", "domain", "host", "scheme", "url_len_bin"]
BOOL_COLUMNS = ["fits_canvas", "truncated", "has_scheme", "is_https", "host_is_ip"]
ITEM_COLUMNS = ["item_id", "cls", "label", "index", "file", "url", "split", "domain", "host", "fits_canvas", "grid",
                "url_len", "url_len_bin", "truncated", "scheme", "has_scheme", "is_https", "host_is_ip",
                "n_host_labels", "path_len", "digit_ratio"] + IMAGE_COLUMNS


def load_items(proc_dir: Path | str, split: str | None = None, cls: str | None = None,
               fits_canvas_only: bool = False) -> pd.DataFrame:
    """Read items.csv.gz and optionally keep one split, one class, or only codes that fit the canvas.

    The image models pass fits_canvas_only=True. The URL models use every row. Text columns keep empty strings
    instead of NaN, and the image columns are NaN until the `images` stage has run.
    """
    t = pd.read_csv(Path(proc_dir) / ITEMS_FILE, keep_default_na=False,
                    na_values={c: [""] for c in IMAGE_COLUMNS + ["grid"]},
                    dtype={c: "string" for c in STRING_COLUMNS} | {c: "bool" for c in BOOL_COLUMNS})
    for c in STRING_COLUMNS:
        t[c] = t[c].astype(object)
    if split is not None:
        t = t[t["split"] == split]
    if cls is not None:
        t = t[t["cls"] == cls]
    if fits_canvas_only:
        t = t[t["fits_canvas"]]
    return t.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------------------------------
# Stage 1: the items table
# ----------------------------------------------------------------------------------------------------------------------
def build_items(manifest: pd.DataFrame, grid: pd.DataFrame, image_features: pd.DataFrame | None = None,
                fracs=DEFAULT_FRACS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Combine the cleaned manifest, the grouped split, the canvas fit and the image features into the items table."""
    rows = manifest[manifest["keep"]].reset_index(drop=True)
    rows = rows.assign(split_id=-1)
    for cls in ["benign", "malicious"]:                        # each class is split on its own, same rule as splits.py
        m = rows["cls"] == cls
        rows.loc[m, "split_id"] = grouped_split(rows.loc[m, "domain"], fracs, seed)
    rows["split"] = rows["split_id"].map(dict(enumerate(SPLIT_NAMES)))
    rows["item_id"] = np.arange(len(rows))

    g = grid[["cls", "file", "grid", "fits_canvas"]]
    rows = rows.merge(g, on=["cls", "file"], how="left", validate="one_to_one")
    if not rows["fits_canvas"].isin([True, False]).all():
        raise ValueError("some kept rows have no measured image. Run: python -m src.canvas_exclusion_count")
    rows["fits_canvas"] = rows["fits_canvas"].astype(bool)

    rows["url_len_bin"] = pd.cut(rows["url_len"], URL_LEN_EDGES, labels=URL_LEN_LABELS).astype(object)
    rows["truncated"] = rows["url_len"] > MAX_URL_LEN

    if image_features is None:
        for c in IMAGE_COLUMNS:
            rows[c] = np.nan
    else:
        f = image_features[image_features["error"].fillna("").eq("")][["cls", "file", "side", "modules", "border", "dark_frac"]]
        rows = rows.merge(f, on=["cls", "file"], how="left", validate="one_to_one")
        rows["qr_version"] = np.where(rows["modules"].isna(), np.nan,
                                      np.where(rows["modules"] >= 21, (rows["modules"] - 17) // 4, 0))
    return rows[ITEM_COLUMNS]


def check_items(items: pd.DataFrame) -> dict:
    """Checks that must hold before the table is saved. Returns the counts used in items_summary.json."""
    assert items["item_id"].is_unique and (items["item_id"].to_numpy() == np.arange(len(items))).all()
    assert not items.duplicated(["cls", "file"]).any()
    assert items["split"].isin(SPLIT_NAMES).all()
    assert (items["label"] == (items["cls"] == "malicious").astype(int)).all()
    for cls, g in items.groupby("cls"):                        # a domain never sits in two splits of the same class
        assert (g.groupby("domain")["split"].nunique() <= 1).all(), f"domain split across splits in {cls}"
    assert not items.loc[items["fits_canvas"], "grid"].isna().any()
    return {s: {c: int(n) for c, n in g["cls"].value_counts().items()} for s, g in items.groupby("split")}


def data_version(items: pd.DataFrame) -> str:
    """Short hash of which file is in which split. It changes if the rows, the order or the split change."""
    h = hashlib.sha1()
    h.update("\n".join(items["cls"] + "/" + items["file"] + "/" + items["split"]).encode())
    return h.hexdigest()[:12]


def run_items(root: Path) -> None:
    proc = root / "data" / "processed"
    manifest = pd.read_csv(proc / "manifest.csv.gz", keep_default_na=False)
    grid_path = proc / "image_grid_sizes.csv.gz"
    if not grid_path.exists():
        raise FileNotFoundError("image_grid_sizes.csv.gz not found. Run: python -m src.canvas_exclusion_count")
    grid = pd.read_csv(grid_path, usecols=["cls", "file", "grid", "fits_canvas"], keep_default_na=False,
                       na_values={"grid": [""]})
    grid["fits_canvas"] = grid["fits_canvas"].astype(str).str.lower().eq("true")
    feat_path = proc / IMAGE_FEATURES_FILE
    features = pd.read_csv(feat_path, keep_default_na=False, na_values={c: [""] for c in ["side", "modules", "border", "dark_frac"]}) \
        if feat_path.exists() else None

    items = build_items(manifest, grid, features)
    counts = check_items(items)
    items.to_csv(proc / ITEMS_FILE, index=False)

    summary = {
        "rows": int(len(items)), "data_version": data_version(items),
        "split_rule": {"key": "domain", "fractions": DEFAULT_FRACS, "seed": DEFAULT_SEED},
        "canvas": DEFAULT_CANVAS, "max_url_len": MAX_URL_LEN, "url_len_bins": URL_LEN_LABELS,
        "rows_by_split_and_class": counts,
        "outside_canvas_by_class": {c: int((~g["fits_canvas"]).sum()) for c, g in items.groupby("cls")},
        "truncated_by_class": {c: int(g["truncated"].sum()) for c, g in items.groupby("cls")},
        "image_features_filled": int(items["modules"].notna().sum()),
    }
    (proc / "items_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved {proc / ITEMS_FILE}: {len(items):,} items, data version {summary['data_version']}, "
          f"image features filled for {summary['image_features_filled']:,} items")
    print("saved", proc / "items_summary.json")


# ----------------------------------------------------------------------------------------------------------------------
# Stage 2: image features for every kept image
# ----------------------------------------------------------------------------------------------------------------------
def measure_image(path: Path) -> dict:
    """Modules per side, quiet zone and dark fraction of one QR PNG, with the same method as 1_eda.ipynb section 4.

    The module size comes from the top row of the top-left finder pattern, which is 7 modules wide. The code area runs
    from the first dark pixel to the last dark column. Anything unreadable is recorded in `error` instead of stopping.
    """
    out = {"side": np.nan, "modules": np.nan, "border": np.nan, "dark_frac": np.nan, "error": ""}
    try:
        with open(path, "rb") as fh:
            a = np.array(Image.open(io.BytesIO(fh.read())))
        dark = ~a if a.dtype == bool else (a < 128)
        h, w = dark.shape[:2]
        ys, xs = np.where(dark)
        y0, x0, x1 = ys.min(), xs.min(), xs.max() + 1
        run = 1
        while x0 + run < w and dark[y0, x0 + run]:
            run += 1
        box = run / 7
        out.update(side=w, modules=round((x1 - x0) / box), border=x0 / box, dark_frac=float(dark.mean()))
        if h != w:
            out["error"] = "not_square"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def run_images(root: Path, workers: int = 8, chunk_size: int = 2000, max_seconds: float | None = None,
               limit_per_class: int | None = None) -> None:
    """Measure every kept image, keeping progress so a stopped run continues where it left off."""
    proc = root / "data" / "processed"
    raw = root / "data" / "raw"
    manifest = pd.read_csv(proc / "manifest.csv.gz", usecols=["cls", "file", "keep"], keep_default_na=False)
    wanted = manifest[manifest["keep"]][["cls", "file"]].reset_index(drop=True)
    if limit_per_class:
        wanted = wanted.groupby("cls").head(limit_per_class)
    partial = proc / IMAGE_FEATURES_PARTIAL
    done = pd.read_csv(partial, keep_default_na=False) if partial.exists() else None
    finished = set(zip(done["cls"], done["file"])) if done is not None else set()
    todo = wanted[[(c, f) not in finished for c, f in zip(wanted["cls"], wanted["file"])]]
    print(f"{len(wanted):,} images, {len(finished):,} already measured, {len(todo):,} to do")

    start, first = time.time(), done is None
    for lo in range(0, len(todo), chunk_size):
        if max_seconds is not None and time.time() - start > max_seconds:
            print(f"time budget reached after {time.time() - start:.0f} s. Run the same command again to continue.")
            return
        chunk = todo.iloc[lo:lo + chunk_size]
        paths = [raw / CLASS_FOLDER[c] / "qrs" / f for c, f in zip(chunk["cls"], chunk["file"])]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(measure_image, paths))
        chunk.reset_index(drop=True).join(pd.DataFrame(res)).to_csv(partial, mode="w" if first else "a", header=first, index=False)
        first = False
        print(f"  measured {lo + len(chunk):,} of {len(todo):,} in {time.time() - start:.0f} s", flush=True)

    table = pd.read_csv(partial, keep_default_na=False, na_values={c: [""] for c in ["side", "modules", "border", "dark_frac"]})
    table.to_csv(proc / IMAGE_FEATURES_FILE, index=False)
    partial.unlink()
    bad = table["error"].fillna("").ne("")
    print(f"saved {proc / IMAGE_FEATURES_FILE} ({len(table):,} images, {int(bad.sum()):,} with an error)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the shared items table (see the top of this file for the schema).")
    p.add_argument("stage", choices=["images", "items"])
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project root (the folder that contains src and data)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=None, help="images stage: stop after this many seconds, rerun to continue")
    p.add_argument("--limit-per-class", type=int, default=None, help="images stage: only the first N images per class (quick test)")
    a = p.parse_args()
    if a.stage == "images":
        run_images(a.root, a.workers, max_seconds=a.max_seconds, limit_per_class=a.limit_per_class)
    else:
        run_items(a.root)


if __name__ == "__main__":
    main()
