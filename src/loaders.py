"""Turn an item into the array a model reads, the same way for everyone.

Image models read a QR code as a 65 x 65 grid of modules (decided in 1_eda.ipynb section 4):
    - one value per module (1 = dark, 0 = light), not one per pixel, because the PNGs are exactly 6 pixels per module
      (1_eda.ipynb checked that the grid reproduces the image exactly),
    - the grid includes the quiet zone (the light margin around the code),
    - a smaller code sits in the top-left corner and the rest of the canvas is light (0),
    - a code whose grid is larger than 65 x 65 is not cropped. It has fits_canvas False in the items table and is left out
      of the image models, so `load_qr_grid` raises an error for it.

URL models read a URL as 256 integers (decided in 1_eda.ipynb section 7):
    - id 0 is padding, id 1 is the unknown token, ids 2 to 96 are the 95 printable ASCII symbols (space to tilde),
    - a URL longer than 256 characters is cut off at 256 (the `truncated` column of the items table marks these),
    - a shorter URL is padded with 0 at the end. The model must mask the padding in its loss.
    - No start or end token is added. A sequence model that needs them adds its own and says so in its notebook.

Typical use:

    grids = load_qr_grids(raw_dir, items_subset)        # uint8 array, shape (n, 65, 65)
    ids = encode_urls(items_subset["url"])               # int16 array, shape (n, 256)
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.canvas_exclusion_count import CLASS_FOLDER, DEFAULT_CANVAS, DEFAULT_MODULE_PX

CANVAS = DEFAULT_CANVAS
MODULE_PX = DEFAULT_MODULE_PX
MAX_URL_LEN = 256
PAD_ID, UNK_ID = 0, 1
FIRST_CHAR_ID = 2
CHARS = "".join(chr(c) for c in range(32, 127))              # the 95 printable ASCII symbols
VOCAB_SIZE = FIRST_CHAR_ID + len(CHARS)                      # 97 ids in total
_ID_OF = {ch: FIRST_CHAR_ID + i for i, ch in enumerate(CHARS)}


def image_path(raw_dir: Path | str, cls: str, file: str) -> Path:
    """Location of one QR PNG under data/raw."""
    return Path(raw_dir) / CLASS_FOLDER[cls] / "qrs" / file


def load_qr_grid(raw_dir: Path | str, cls: str, file: str, canvas: int = CANVAS, module_px: int = MODULE_PX) -> np.ndarray:
    """One QR code as a (canvas, canvas) uint8 array, 1 for a dark module, 0 for a light one, top-left anchored."""
    with open(image_path(raw_dir, cls, file), "rb") as fh:
        a = np.array(Image.open(io.BytesIO(fh.read())))
    dark = ~a if a.dtype == bool else (a < 128)
    h, w = dark.shape[:2]
    if h != w or w % module_px:
        raise ValueError(f"{file}: image is {h} x {w} pixels, not a square multiple of {module_px}")
    grid = dark[module_px // 2::module_px, module_px // 2::module_px]
    if grid.shape[0] > canvas:
        raise ValueError(f"{file}: grid is {grid.shape[0]} x {grid.shape[0]} modules, larger than the {canvas} x {canvas} canvas")
    out = np.zeros((canvas, canvas), dtype=np.uint8)
    out[:grid.shape[0], :grid.shape[1]] = grid
    return out


def load_qr_grids(raw_dir: Path | str, items: pd.DataFrame, workers: int = 8) -> np.ndarray:
    """The grids of many items (rows of the items table) as one uint8 array of shape (n, canvas, canvas)."""
    pairs = list(zip(items["cls"], items["file"]))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        grids = list(ex.map(lambda p: load_qr_grid(raw_dir, p[0], p[1]), pairs))
    return np.stack(grids) if grids else np.zeros((0, CANVAS, CANVAS), dtype=np.uint8)


def cached_grids(raw_dir: Path | str, items: pd.DataFrame, cache_path: Path | str, workers: int = 8) -> np.ndarray:
    """Like load_qr_grids, but the first call saves the grids to cache_path (a .npz file) and later calls read it.

    Reading hundreds of thousands of PNGs is slow, so do it once per split. The cache stores the item ids it was built
    for and is rebuilt if the items differ.
    """
    cache_path = Path(cache_path)
    ids = items["item_id"].to_numpy()
    if cache_path.exists():
        with np.load(cache_path) as z:
            if np.array_equal(z["item_id"], ids):
                return z["grids"]
    grids = load_qr_grids(raw_dir, items, workers)
    np.savez(cache_path, item_id=ids, grids=grids)
    return grids


def encode_url(url: str, max_len: int = MAX_URL_LEN) -> np.ndarray:
    """One URL as an int16 array of length max_len (ids described at the top of this file)."""
    out = np.full(max_len, PAD_ID, dtype=np.int16)
    for i, ch in enumerate(url[:max_len]):
        out[i] = _ID_OF.get(ch, UNK_ID)
    return out


def encode_urls(urls, max_len: int = MAX_URL_LEN) -> np.ndarray:
    """Many URLs as one int16 array of shape (n, max_len)."""
    urls = list(urls)
    return np.stack([encode_url(u, max_len) for u in urls]) if urls else np.zeros((0, max_len), dtype=np.int16)


def decode_url(ids) -> str:
    """Turn ids back into text. Padding is dropped and the unknown token is shown as a question mark."""
    return "".join(CHARS[i - FIRST_CHAR_ID] if i >= FIRST_CHAR_ID else "?" for i in np.asarray(ids).tolist() if i != PAD_ID)
