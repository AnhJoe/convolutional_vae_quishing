"""Floor baselines: a Gaussian fitted to benign training items, scored by Mahalanobis distance (EDA section 7).

With one feature the score is the absolute z-score. Higher means more likely malicious, the same convention as the VAE
scores. A small ridge keeps the covariance invertible, since host_is_ip is almost constant among benign URLs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class GaussianFloor:
    def __init__(self, features: list[str], ridge: float = 1e-3):
        self.features = list(features)
        self.ridge = ridge

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy(dtype=np.float64)

    def fit(self, benign_train: pd.DataFrame) -> "GaussianFloor":
        x = self._x(benign_train)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        z = (x - self.mean_) / self.scale_                       # standardise first, so one ridge value suits every feature
        cov = np.atleast_2d(np.cov(z, rowvar=False)) + self.ridge * np.eye(len(self.features))
        self.prec_ = np.linalg.inv(cov)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        z = (self._x(df) - self.mean_) / self.scale_
        return np.sqrt(np.einsum("ij,jk,ik->i", z, self.prec_, z))
