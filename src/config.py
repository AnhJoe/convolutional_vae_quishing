"""Paths and constants shared by every step after the EDA.

The constants restate decisions made in notebooks/1_eda.ipynb, so the models and the evaluation read them from one place.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
SCORES_DIR = RESULTS_DIR / "scores"
CKPT_DIR = RESULTS_DIR / "checkpoints"

CLASS_FOLDERS = {"benign": "QR_All_benign", "malicious": "QR_All_Malicious"}

# URL model input (EDA section 7): 95 printable ASCII ids + unknown + padding, cut off at 256 characters
URL_MAX_LEN = 256
PAD_ID, UNK_ID = 0, 1
VOCAB_SIZE = 2 + 95

# QR image input (EDA section 4): 6 px per module, one pixel per module, 65 x 65 canvas anchored top-left
MODULE_PX = 6
CANVAS = 65

# Floors (EDA section 7): benign-only Gaussian, Mahalanobis distance
FLOORS = {
    "A": {"image": ["modules", "dark_frac"], "url": ["url_len"]},
    "B": {"image": ["modules", "dark_frac"],
          "url": ["url_len", "has_scheme", "host_is_ip", "n_host_labels", "path_len", "digit_ratio"]},
}

# Evaluation
FPR_TARGETS = (0.01, 0.05)
N_BOOT = 1000
MIN_PER_CLASS_STRATUM = 200     # a stratum needs this many items of each class for a stable AUROC
URL_LEN_BIN_EDGES = (0, 20, 30, 40, 60, 100, 10**9)


def find_raw_dir() -> Path:
    """The folder holding QR_All_benign and QR_All_Malicious: data/raw (where 1_eda.ipynb downloads them), else the repo root."""
    for cand in (ROOT / "data" / "raw", ROOT):
        if all((cand / f).is_dir() for f in CLASS_FOLDERS.values()):
            return cand
    raise FileNotFoundError(f"QR_All_benign and QR_All_Malicious not found in {ROOT / 'data' / 'raw'} or {ROOT}")
