"""Checks for the RQ1 bootstrap scorer and the case sample used in the error audit."""
import unittest

import numpy as np
from sklearn.metrics import f1_score, hamming_loss

from src.rq1_audit import _resampled_scores


class BootstrapScorerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random((200, 6)) < 0.3).astype(np.int8)
        self.pred = (rng.random((200, 6)) < 0.3).astype(np.int8)
        self.pred[:, 5] = 0  # a label never predicted: sklearn zero_division=0 must be matched
        self.mask = np.ones(6, bool)

    def test_unit_weights_match_sklearn(self):
        s = _resampled_scores(np.ones((1, 200), dtype=np.int64), self.y, self.pred, self.mask)
        self.assertAlmostEqual(s["macro_f1"][0], f1_score(self.y, self.pred, average="macro", zero_division=0))
        self.assertAlmostEqual(s["micro_f1"][0], f1_score(self.y, self.pred, average="micro", zero_division=0))
        self.assertAlmostEqual(s["hamming_loss"][0], hamming_loss(self.y, self.pred))

    def test_weights_equal_explicit_resample(self):
        rng = np.random.default_rng(1)
        idx = rng.integers(0, 200, 200)
        w = np.bincount(idx, minlength=200)[None, :]
        s = _resampled_scores(w, self.y, self.pred, self.mask)
        y, p = self.y[idx], self.pred[idx]
        self.assertAlmostEqual(s["macro_f1"][0], f1_score(y, p, average="macro", zero_division=0))
        self.assertAlmostEqual(s["hamming_loss"][0], hamming_loss(y, p))

    def test_label_subset(self):
        mask = np.array([True, False, True, False, False, False])
        s = _resampled_scores(np.ones((1, 200), dtype=np.int64), self.y, self.pred, mask)
        self.assertAlmostEqual(s["macro_f1"][0],
                               f1_score(self.y[:, mask], self.pred[:, mask], average="macro", zero_division=0))


if __name__ == "__main__":
    unittest.main()
