from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.simulation.recourse import ActionableFeatureSpec, ScenarioSimulator


def toy_score(frame: pd.DataFrame) -> list[float]:
    row = frame.iloc[0]
    return [float(6.0 + row["studytime"] - 0.5 * row["absences"] + 1.5 * (row["schoolsup"] == "yes"))]


class RecourseTests(unittest.TestCase):
    def test_rank_scenarios_prefers_positive_changes(self) -> None:
        simulator = ScenarioSimulator(
            score_fn=toy_score,
            action_space={
                "studytime": ActionableFeatureSpec(
                    feature="studytime",
                    candidate_values=(2, 3, 4),
                    cost=1.0,
                    rationale="Increase structured study support.",
                ),
                "schoolsup": ActionableFeatureSpec(
                    feature="schoolsup",
                    candidate_values=("yes",),
                    cost=0.8,
                    rationale="Add school support.",
                ),
            },
            target_grade=10.0,
            top_k=3,
            max_feature_changes=2,
        )

        student = pd.Series({"studytime": 1, "absences": 4, "schoolsup": "no"})
        scenarios = simulator.rank_scenarios(student)

        self.assertGreater(len(scenarios), 0)
        self.assertGreater(scenarios[0].predicted_grade, 5.0)
        self.assertTrue(any(change.feature == "schoolsup" for change in scenarios[0].changes))


if __name__ == "__main__":
    unittest.main()
