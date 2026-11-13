"""Floor baselines: a Gaussian fitted on benign train rows only, scored by its distance (1_eda.ipynb section 7).

    floorA_image / mahalanobis   modules, dark_frac                 (image floor B is the same, so it is not saved twice)
    floorA_url   / abs_z         url_len                            (one feature: the absolute z-score)
    floorB_url   / mahalanobis   url_len, has_scheme, host_is_ip, n_host_labels, path_len, digit_ratio

Higher means more likely malicious, the convention of src/evaluate.py. The features are standardised on benign train
rows and a small ridge is added to their covariance, because host_is_ip is almost constant among benign rows.

    from src.floors import run_floors
    metrics = run_floors(RESULTS_DIR, items, "canvas")    # image floor, items = the codes that fit the canvas
    metrics = run_floors(RESULTS_DIR, items, "all")       # URL floors A and B
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import save_and_evaluate

FLOORS = {
    "canvas": [("floorA_image", "mahalanobis", ["modules", "dark_frac"])],
    "all": [("floorA_url", "abs_z", ["url_len"]),
            ("floorB_url", "mahalanobis", ["url_len", "has_scheme", "host_is_ip", "n_host_labels", "path_len", "digit_ratio"])],
}


class GaussianFloor:
    """Mahalanobis distance from a Gaussian fitted on benign train rows. With one feature it is the absolute z-score."""

    def __init__(self, features: list[str], ridge: float = 1e-3):
        self.features, self.ridge = list(features), ridge

    def _z(self, df: pd.DataFrame) -> np.ndarray:
        return (df[self.features].to_numpy(dtype=np.float64) - self.mean_) / self.scale_

    def fit(self, benign_train: pd.DataFrame) -> "GaussianFloor":
        x = benign_train[self.features].to_numpy(dtype=np.float64)
        if np.isnan(x).any():
            raise ValueError(f"floor features {self.features} have missing values in the benign train rows")
        self.mean_, self.scale_ = x.mean(axis=0), x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        z = (x - self.mean_) / self.scale_
        cov = np.atleast_2d(np.cov(z, rowvar=False)) + self.ridge * np.eye(len(self.features))
        self.precision_ = np.linalg.inv(cov)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        z = self._z(df)
        return np.sqrt(np.einsum("ij,jk,ik->i", z, self.precision_, z))


def run_floors(results_dir: Path | str, items: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Fit every floor of the scope on benign train rows, save validation and test scores, return the metrics rows.

    `items` is the full items table. For scope "canvas" only the codes that fit the canvas are used, as for the image
    models.
    """
    if scope == "canvas":
        items = items[items["fits_canvas"]]
    benign_train = items[(items["split"] == "train") & (items["cls"] == "benign")]
    val, test = items[items["split"] == "validation"], items[items["split"] == "test"]
    rows = []
    for model, score, features in FLOORS[scope]:
        floor = GaussianFloor(features).fit(benign_train)
        rows.append(save_and_evaluate(results_dir, items, scope, model, score, 0,
                                      validation=(val["item_id"], floor.score(val)), test=(test["item_id"], floor.score(test))))
    return pd.concat(rows, ignore_index=True)
