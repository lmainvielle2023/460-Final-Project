"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float
    pass_fail_accuracy: float
    pass_fail_precision: float
    pass_fail_recall: float
    pass_fail_f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "pass_fail_accuracy": self.pass_fail_accuracy,
            "pass_fail_precision": self.pass_fail_precision,
            "pass_fail_recall": self.pass_fail_recall,
            "pass_fail_f1": self.pass_fail_f1,
        }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pass_threshold: float = 10.0,
) -> RegressionMetrics:
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))

    baseline = np.mean(y_true)
    total_sum_squares = float(np.sum(np.square(y_true - baseline)))
    residual_sum_squares = float(np.sum(np.square(residual)))
    r2 = 1.0 - (residual_sum_squares / total_sum_squares) if total_sum_squares else 0.0

    true_pass = y_true >= pass_threshold
    pred_pass = y_pred >= pass_threshold
    pass_fail_accuracy = float(np.mean(true_pass == pred_pass))
    true_positive = float(np.sum(true_pass & pred_pass))
    false_positive = float(np.sum(~true_pass & pred_pass))
    false_negative = float(np.sum(true_pass & ~pred_pass))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return RegressionMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        pass_fail_accuracy=pass_fail_accuracy,
        pass_fail_precision=float(precision),
        pass_fail_recall=float(recall),
        pass_fail_f1=float(f1),
    )
