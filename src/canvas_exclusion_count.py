"""Count the QR images that do not fit the model's fixed canvas.

The image VAE takes one input size (a CANVAS x CANVAS grid of modules, the black or white squares of a QR code).
A code whose full grid (modules plus quiet zone) is larger than the canvas cannot be placed on it without cropping,
so it is excluded from the image model and counted. This script measures that count over all kept images (the rows of
the manifest with keep true, the same rows as the items table).

How it works. Every image is drawn at a fixed number of pixels per module, so the grid size is the PNG width divided
by that number. The width is stored in the first 24 bytes of a PNG file, so only those bytes are read instead of
decoding the image. This makes a pass over about one million files practical.

2_data_processing.ipynb calls `run` from this file. It can also be run from the project root:

    python -m src.canvas_exclusion_count

Outputs, written to data/processed/:
    image_grid_sizes.csv.gz        one row per image: cls, file, width, height, status, grid, fits_canvas
    canvas_exclusion_summary.json  counts per class, and the share of images that fit other canvas sizes

A run can be stopped and started again: progress is kept in image_grid_sizes_partial.csv and finished files are
skipped. Use --max-seconds to stop politely after a time budget (for example in a time-limited notebook cell).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CLASS_FOLDER = {"benign": "QR_All_benign", "malicious": "QR_All_Malicious"}   # folder names under data/raw
DEFAULT_CANVAS = 65                                                            # decided in 1_eda.ipynb section 4
DEFAULT_MODULE_PX = 6                                                          # pixels per module, measured in 1_eda.ipynb
COVERAGE_CANVASES = [37, 41, 49, 53, 57, 65, 77, 85]                           # other sizes, for comparison with the EDA


def read_png_size(path: Path) -> tuple[str, int, int]:
    """Return (status, width, height) from the PNG header. Status is 'ok' or the reason the file could not be read."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(24)
    except FileNotFoundError:
        return "missing_file", 0, 0
    except OSError:
        return "unreadable_file", 0, 0
    if len(head) < 24 or head[:8] != PNG_SIGNATURE or head[12:16] != b"IHDR":
        return "not_png", 0, 0
    width, height = struct.unpack(">II", head[16:24])
    return "ok", width, height


def measure_images(manifest: pd.DataFrame, raw_dir: Path, partial_path: Path, workers: int = 16,
                   chunk_size: int = 5000, max_seconds: float | None = None) -> pd.DataFrame | None:
    """Read the PNG size of every image in the manifest, resuming from partial_path.
    Returns the full table when every image is done, or None if the time budget ran out first."""
    wanted = manifest[["cls", "file"]].drop_duplicates().reset_index(drop=True)
    if partial_path.exists():
        done = pd.read_csv(partial_path, keep_default_na=False)
    else:
        done = pd.DataFrame(columns=["cls", "file", "status", "width", "height"])
    finished = set(zip(done["cls"], done["file"]))
    todo = wanted[[(c, f) not in finished for c, f in zip(wanted["cls"], wanted["file"])]]
    print(f"{len(wanted):,} images, {len(wanted) - len(todo):,} already measured, {len(todo):,} to do")

    start = time.time()
    first_write = not partial_path.exists()
    for lo in range(0, len(todo), chunk_size):
        if max_seconds is not None and time.time() - start > max_seconds:
            print(f"time budget reached after {time.time() - start:.0f} s. Run the same command again to continue.")
            return None
        chunk = todo.iloc[lo:lo + chunk_size]
        paths = [raw_dir / CLASS_FOLDER[c] / "qrs" / f for c, f in zip(chunk["cls"], chunk["file"])]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            sizes = list(ex.map(read_png_size, paths))
        out = chunk.assign(status=[s[0] for s in sizes], width=[s[1] for s in sizes], height=[s[2] for s in sizes])
        out.to_csv(partial_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False
        print(f"  measured {lo + len(chunk):,} of {len(todo):,} in {time.time() - start:.0f} s", flush=True)
    table = pd.read_csv(partial_path, keep_default_na=False)
    return table.merge(wanted, on=["cls", "file"], how="inner")      # drop rows left over from an earlier run with other rows


def add_grid_and_fit(sizes: pd.DataFrame, module_px: int, canvas: int) -> pd.DataFrame:
    """Add the grid size (modules plus quiet zone per side) and whether the image fits the canvas.
    An image that is not square or whose width is not a multiple of module_px is flagged and counted as not fitting."""
    t = sizes.copy()
    ok = t["status"].eq("ok")
    square = t["width"].eq(t["height"])
    multiple = (t["width"] % module_px).eq(0)
    t.loc[ok & ~square, "status"] = "not_square"
    t.loc[ok & square & ~multiple, "status"] = "not_a_multiple_of_module_size"
    good = t["status"].eq("ok")
    t["grid"] = (t["width"] // module_px).where(good)
    t["fits_canvas"] = good & (t["grid"] <= canvas)
    return t


def summarize(table: pd.DataFrame, module_px: int, canvas: int) -> dict:
    """Counts per class: images, readable images, images outside the canvas, and coverage of other canvas sizes."""
    per_class = {}
    for cls, t in table.groupby("cls"):
        good = t[t["status"].eq("ok")]
        outside = good[good["grid"] > canvas]
        per_class[cls] = {
            "images": int(len(t)),
            "irregular_or_unreadable": {k: int(v) for k, v in t.loc[~t["status"].eq("ok"), "status"].value_counts().items()},
            "outside_canvas": int(len(outside)),
            "outside_canvas_share": round(len(outside) / len(t), 6),
            "largest_grid": int(good["grid"].max()),
            "outside_canvas_by_grid": {int(k): int(v) for k, v in outside["grid"].value_counts().sort_index().items()},
            "share_fitting_other_canvases": {str(g): round(float((good["grid"] <= g).sum() / len(t)), 6) for g in COVERAGE_CANVASES},
        }
    return {"canvas": canvas, "module_px": module_px, "images": int(len(table)),
            "outside_canvas": int(sum(v["outside_canvas"] for v in per_class.values())), "per_class": per_class}


def run(root: Path, canvas: int = DEFAULT_CANVAS, module_px: int | None = None, workers: int = 16,
        max_seconds: float | None = None, limit_per_class: int | None = None, out_dir: Path | None = None) -> dict | None:
    """Measure every kept image in the manifest, save the tables, print the counts and return the summary.

    Returns None if the time budget ran out first. Call it again to continue. Used by 2_data_processing.ipynb and main().
    """
    raw_dir = root / "data" / "raw"
    proc_dir = root / "data" / "processed"
    out_dir = out_dir or proc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if module_px is None:
        summary_path = proc_dir / "eda_summary.json"
        module_px = int(json.loads(summary_path.read_text())["module_size_px"]) if summary_path.exists() else DEFAULT_MODULE_PX

    manifest = pd.read_csv(proc_dir / "manifest.csv.gz", usecols=["cls", "file", "keep"], keep_default_na=False)
    manifest = manifest[manifest["keep"].astype(str).str.lower().eq("true")]      # kept rows only, as in the items table
    if limit_per_class:
        manifest = manifest.groupby("cls").head(limit_per_class)

    partial = out_dir / "image_grid_sizes_partial.csv"
    sizes = measure_images(manifest, raw_dir, partial, workers=workers, max_seconds=max_seconds)
    if sizes is None:
        return None
    table = add_grid_and_fit(sizes, module_px, canvas)
    table.to_csv(out_dir / "image_grid_sizes.csv.gz", index=False)
    summary = summarize(table, module_px, canvas)
    (out_dir / "canvas_exclusion_summary.json").write_text(json.dumps(summary, indent=2))
    partial.unlink()

    print(f"\ncanvas {canvas} x {canvas} modules, {module_px} px per module")
    for cls, s in summary["per_class"].items():
        print(f"{cls}: {s['images']:,} images, {s['outside_canvas']:,} outside the canvas ({100 * s['outside_canvas_share']:.3f}%), "
              f"irregular or unreadable: {s['irregular_or_unreadable'] or 'none'}")
    print("saved", out_dir / "image_grid_sizes.csv.gz", "and", out_dir / "canvas_exclusion_summary.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Count QR images that do not fit the model canvas (reads PNG headers only).")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project root (the folder that contains src and data)")
    p.add_argument("--canvas", type=int, default=DEFAULT_CANVAS, help="canvas side in modules, quiet zone included")
    p.add_argument("--module-px", type=int, default=None, help="pixels per module (default: module_size_px from eda_summary.json, else 6)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-seconds", type=float, default=None, help="stop after this many seconds and resume on the next run")
    p.add_argument("--limit-per-class", type=int, default=None, help="only the first N images per class (for a quick test)")
    p.add_argument("--out-dir", type=Path, default=None, help="where to write outputs (default: <root>/data/processed)")
    a = p.parse_args()
    run(a.root, a.canvas, a.module_px, a.workers, a.max_seconds, a.limit_per_class, a.out_dir)


if __name__ == "__main__":
    main()
