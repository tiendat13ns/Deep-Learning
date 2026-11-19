"""Checks for calibration correctness and the validation/test boundary."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analyze_thresholds import DEFAULT_RUN, _check_ids, calibrate, evaluate
from src.data import LABELS, label_tiers
from src.metrics import compute_metrics
from src.thresholds import calibrate_thresholds


class CalibrationTests(unittest.TestCase):
    def test_two_classes_need_different_thresholds(self):
        # Class A: positive scores 0.4/0.3, negatives 0.2/0.1.
        # Class B: positive scores 0.9/0.8, negatives 0.7/0.6.
        # Shared 0.3 gets class F1 [1, 2/3]; independent [0.3, 0.8] is perfect.
        y = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
        p = np.array([[0.4, 0.9], [0.3, 0.8], [0.2, 0.7], [0.1, 0.6]])
        result = calibrate_thresholds(y, p, [0.3, 0.5, 0.8])
        self.assertEqual(result["global_threshold"], 0.3)
        np.testing.assert_array_equal(result["per_label_thresholds"], [0.3, 0.8])
        self.assertAlmostEqual(compute_metrics(y, p, 0.3)["macro_f1"], 5 / 6)
        self.assertEqual(compute_metrics(y, p, result["per_label_thresholds"])["macro_f1"], 1.0)

    def test_threshold_equality_counts_as_positive(self):
        result = calibrate_thresholds([[1], [0]], [[0.5], [0.4]], [0.4, 0.5, 0.6])
        self.assertEqual(result["global_threshold"], 0.5)
        self.assertEqual(result["per_label_f1_grid"][1, 0], 1.0)

    def test_ties_prefer_nearest_half_then_smaller(self):
        from src.thresholds import _select_best
        self.assertEqual(_select_best(np.array([1, 1, 1]), np.array([0.3, 0.5, 0.8])), 1)
        self.assertEqual(_select_best(np.array([1, 0, 1]), np.array([0.4, 0.5, 0.6])), 0)

    def test_no_positive_validation_class_stays_at_half(self):
        result = calibrate_thresholds([[0], [0]], [[0.8], [0.2]])
        self.assertEqual(result["per_label_thresholds"].item(), 0.5)

    def test_rejects_invalid_probabilities_and_grid(self):
        for p in ([[float("nan")]], [[-0.1]], [[1.1]]):
            with self.assertRaises(ValueError):
                calibrate_thresholds([[1]], p)
        with self.assertRaises(ValueError):
            calibrate_thresholds([[1]], [[0.4]], [0.2, 0.8])
        with self.assertRaises(ValueError):
            calibrate_thresholds([[1]], [[0.4]], [0.5, 0.5])

    def test_per_label_validation_macro_cannot_be_below_global(self):
        rng = np.random.default_rng(42)
        y, p = rng.integers(0, 2, (60, 6)), rng.random((60, 6))
        result = calibrate_thresholds(y, p)
        fixed = compute_metrics(y, p, 0.5)["macro_f1"]
        global_f1 = compute_metrics(y, p, result["global_threshold"])["macro_f1"]
        per_label = compute_metrics(y, p, result["per_label_thresholds"])["macro_f1"]
        self.assertGreaterEqual(per_label + 1e-12, global_f1)
        self.assertGreaterEqual(global_f1 + 1e-12, fixed)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "results" / "rq1"
        for folder in ("results/preds", "results/tables", "splits"):
            (self.root / folder).mkdir(parents=True)
        counts = np.array([1100] * 10 + [850] * 10 + [650] * 8)
        pd.DataFrame({"train": counts}, index=LABELS).to_csv(self.root / "results/tables/label_distribution.csv")
        (self.root / "splits/label_tiers.json").write_text(json.dumps({
            "labels": LABELS, "tiers": label_tiers(counts), "head_min": 1000, "tail_max": 750,
        }), encoding="utf-8")
        pd.DataFrame({"sample_id": ["v1", "v2"]}).to_csv(self.root / "splits/val.csv", index=False)
        np.savez_compressed(self.root / "results/preds" / f"{DEFAULT_RUN}_val.npz",
                            ids=np.array(["v1", "v2"]), y=np.ones((2, 28), dtype=np.int8),
                            p=np.full((2, 28), 0.6))

    def test_calibration_succeeds_without_any_test_data(self):
        spec = calibrate(self.root, self.out)
        self.assertEqual(spec["calibration_split"], "val")
        self.assertTrue((self.out / "calibration.json").exists())
        self.assertFalse((self.root / "results/preds" / f"{DEFAULT_RUN}_test.npz").exists())

    def test_evaluation_requires_existing_frozen_thresholds(self):
        with self.assertRaises(FileNotFoundError):
            evaluate(self.root, self.out)

    def test_changed_validation_source_is_rejected_before_test_load(self):
        calibrate(self.root, self.out)
        (self.root / "splits/val.csv").write_text("sample_id\nv2\nv1\n", encoding="utf-8")
        with patch("src.analyze_thresholds.load_preds") as loader:
            with self.assertRaisesRegex(ValueError, "changed after threshold freeze"):
                evaluate(self.root, self.out)
            loader.assert_not_called()

    def test_misaligned_prediction_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "row order"):
            _check_ids(self.root, "val", np.array(["v2", "v1"]))


if __name__ == "__main__":
    unittest.main()
