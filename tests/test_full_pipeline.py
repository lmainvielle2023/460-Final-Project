from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.pipelines.full import run_all


def _synthetic_student_frame() -> pd.DataFrame:
    rows = []
    for index in range(36):
        rows.append(
            {
                "school": "GP" if index % 2 else "MS",
                "sex": "F" if index % 2 else "M",
                "age": 15 + (index % 4),
                "address": "U",
                "famsize": "GT3",
                "Pstatus": "T",
                "Medu": index % 5,
                "Fedu": (index + 1) % 5,
                "Mjob": "teacher",
                "Fjob": "services",
                "reason": "course",
                "guardian": "mother",
                "traveltime": 1 + (index % 3),
                "studytime": 1 + (index % 4),
                "failures": 1 if index % 9 == 0 else 0,
                "schoolsup": "yes" if index % 5 == 0 else "no",
                "famsup": "yes" if index % 3 == 0 else "no",
                "paid": "no",
                "activities": "yes" if index % 2 else "no",
                "nursery": "yes",
                "higher": "yes",
                "internet": "yes" if index % 4 else "no",
                "romantic": "no",
                "famrel": 3 + (index % 3),
                "freetime": 2 + (index % 3),
                "goout": 2 + (index % 3),
                "Dalc": 1,
                "Walc": 1 + (index % 3),
                "health": 3,
                "absences": index % 8,
                "G1": 7 + (index % 8),
                "G2": 8 + (index % 8),
                "G3": 8 + (index % 8),
            }
        )
    return pd.DataFrame(rows)


class FullPipelineTests(unittest.TestCase):
    def test_run_all_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            _synthetic_student_frame().to_csv(raw_dir / "student-por.csv", sep=";", index=False)
            config_path = root / "config.toml"
            config_path.write_text(
                f"""
[project]
name = "test"
random_seed = 42

[dataset]
raw_dir = "{raw_dir}"
subject = "por"
target = "G3"
include_prior_grades = false
pass_threshold = 10.0

[features]
generate_interactions = true
interaction_pairs = ["studytime*absences", "studytime*failures"]

[modeling]
candidate_models = ["linear_regression", "ridge", "decision_tree"]
interpretable_models = ["linear_regression", "ridge", "decision_tree"]
test_size = 0.2
cross_validation_folds = 3
max_mlp_iter = 200

[simulation]
target_grade = 10.0
top_k = 2
max_feature_changes = 1
sample_size = 3

[intervention]
planner_mode = "rule_based"
include_agent_prompt = false

[outputs]
reports_dir = "{root / 'reports'}"
figures_dir = "{root / 'reports' / 'figures'}"
models_dir = "{root / 'models'}"
selected_model_path = "{root / 'models' / 'selected.joblib'}"
""",
                encoding="utf-8",
            )

            paths = run_all(config_path)

            self.assertTrue(paths["metrics"].exists())
            self.assertTrue(paths["cross_validation_metrics"].exists())
            self.assertTrue(paths["scenario_summary"].exists())
            self.assertTrue(paths["final_results_summary"].exists())


if __name__ == "__main__":
    unittest.main()
