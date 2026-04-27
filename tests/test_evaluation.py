from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.models.evaluation import compute_regression_metrics


class EvaluationTests(unittest.TestCase):
    def test_regression_and_pass_fail_metrics_are_computed(self) -> None:
        metrics = compute_regression_metrics(
            y_true=np.array([8.0, 9.0, 11.0, 13.0]),
            y_pred=np.array([7.0, 10.5, 10.0, 12.0]),
            pass_threshold=10.0,
        )

        self.assertGreater(metrics.rmse, 0.0)
        self.assertEqual(metrics.pass_fail_accuracy, 0.75)
        self.assertAlmostEqual(metrics.pass_fail_precision, 2 / 3)
        self.assertAlmostEqual(metrics.pass_fail_recall, 1.0)
        self.assertAlmostEqual(metrics.pass_fail_f1, 0.8)


if __name__ == "__main__":
    unittest.main()
