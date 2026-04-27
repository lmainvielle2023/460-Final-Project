from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.features.engineering import InteractionSpec, build_interaction_features


class EngineeringTests(unittest.TestCase):
    def test_build_interaction_features_adds_expected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "studytime": [1, 2],
                "absences": [3, 4],
                "schoolsup": ["yes", "no"],
            }
        )

        enriched = build_interaction_features(
            frame,
            [
                InteractionSpec("studytime", "absences"),
                InteractionSpec("studytime", "schoolsup"),
            ],
        )

        self.assertEqual(enriched.loc[1, "studytime_x_absences"], 8.0)
        self.assertEqual(enriched.loc[0, "studytime_x_schoolsup"], 1.0)
        self.assertEqual(enriched.loc[1, "studytime_x_schoolsup"], 0.0)


if __name__ == "__main__":
    unittest.main()
