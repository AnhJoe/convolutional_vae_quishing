"""Regression checks for domain isolation and tied-score operating points."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.url_data import prepare, url_features
from src.url_eval import RankedScores, calibrate


class URLProtocolTest(unittest.TestCase):
    def test_shared_domains_stay_together_across_classes(self):
        with TemporaryDirectory() as directory:
            benign = Path(directory) / "benign.csv"
            malicious = Path(directory) / "malicious.csv"
            benign_urls = [f"https://a.site{i}.co.uk/benign" for i in range(40)]
            malicious_urls = [f"b.site{i}.co.uk/malicious" for i in range(40)]
            collision = "https://example.com/exact"
            pd.DataFrame({"url": [*benign_urls, benign_urls[0], collision, " "]}).to_csv(benign, index=False)
            pd.DataFrame({"url": [*malicious_urls, collision]}).to_csv(malicious, index=False)
            rows, audit = prepare(benign, malicious, split_seed=0)
        self.assertEqual(len(rows), 80)
        self.assertTrue(rows.groupby("domain").split.nunique().eq(1).all())
        self.assertEqual(audit["cross_split_domains"], 0)
        self.assertEqual(audit["conflicting_rows_removed"], 2)
        self.assertEqual(audit["same_class_duplicates_removed"], 1)
        self.assertEqual(audit["blank_rows_removed"], 1)

    def test_private_suffix_and_canonical_ip(self):
        self.assertEqual(url_features("https://a.user.github.io/x")[0], "dns:user.github.io")
        self.assertEqual(url_features("https://[::ffff:192.0.2.1]/")[0], "ip:192.0.2.1")

    def test_tied_scores_and_strict_thresholds(self):
        scores = np.array([1.0, 1.0, 0.0, 2.0])
        labels = np.array([0, 1, 0, 1])
        thresholds = calibrate(scores, labels == 0)
        self.assertEqual(thresholds, [1.0, 1.0])
        metrics = RankedScores(scores, labels, thresholds).metrics(np.ones(4, dtype=int))
        np.testing.assert_allclose(metrics, [0.875, 5 / 6, 0.5, 0, 0.5, 0])


if __name__ == "__main__":
    unittest.main()
