from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.features.engineering import InteractionSpec
from student_success.models.training import prepare_model_frame


class TrainingTests(unittest.TestCase):
    def test_prepare_model_frame_can_exclude_prior_grades(self) -> None:
        frame = pd.DataFrame(
            {
                "studytime": [1, 2],
                "absences": [0, 2],
                "G1": [8, 9],
                "G2": [9, 10],
                "G3": [10, 11],
            }
        )

        x, y = prepare_model_frame(
            frame,
            include_prior_grades=False,
            interaction_specs=[InteractionSpec("studytime", "absences")],
        )

        self.assertNotIn("G1", x.columns)
        self.assertNotIn("G2", x.columns)
        self.assertIn("studytime_x_absences", x.columns)
        self.assertEqual(y.tolist(), [10, 11])


if __name__ == "__main__":
    unittest.main()
