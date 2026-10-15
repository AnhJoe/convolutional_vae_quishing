"""Train, validation and test splits grouped by domain.

The rule is decided in notebooks/1_eda.ipynb section 6 and applied in notebooks/2_data_processing.ipynb (through
src/prepare_data.py): whole domains go to one split, the largest domains fill train first, and the remaining domains are
shuffled and dealt to validation and test in the ratio of their fractions. Benign and malicious rows are split separately
with the same fractions, so every split has about the same share of malicious rows.

The split of every item is stored in the `split` column of the items table, read with `load_items` (src/prepare_data.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_NAMES = ["train", "validation", "test"]
DEFAULT_FRACS = [0.70, 0.15, 0.15]
DEFAULT_SEED = 0


def grouped_split(groups, fracs=DEFAULT_FRACS, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Return a split id (0 = train, 1 = validation, 2 = test) for every row, keeping each group in one split.

    The largest groups fill train first, up to its share of the rows. The groups left over are shuffled with the seed
    and dealt to validation and test in the ratio of their fractions.
    """
    codes, uniq = pd.factorize(pd.Series(groups))
    sizes = np.bincount(codes)
    by_size = np.argsort(-sizes, kind="stable")
    split_of_group = np.zeros(len(uniq), dtype=int)
    in_train = np.zeros(len(uniq), dtype=bool)
    in_train[by_size[np.cumsum(sizes[by_size]) <= fracs[0] * sizes.sum()]] = True
    rest = np.random.default_rng(seed).permutation(np.flatnonzero(~in_train))
    frac_val = fracs[1] / (fracs[1] + fracs[2])
    split_of_group[rest] = 1 + (np.cumsum(sizes[rest]) / sizes[rest].sum() > frac_val).astype(int)
    return split_of_group[codes]
