from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_success.interventions.planner import RuleBasedInterventionPlanner
from student_success.schemas import PredictionRecord, ScenarioChange, SimulationResult


class PlannerTests(unittest.TestCase):
    def test_rule_based_plan_uses_best_scenario(self) -> None:
        scenario = SimulationResult(
            predicted_grade=10.5,
            grade_delta=2.0,
            feasibility_score=1.0,
            meets_target=True,
            changes=[
                ScenarioChange(
                    feature="studytime",
                    old_value=1,
                    new_value=3,
                    rationale="Increase structured study support.",
                )
            ],
        )
        planner = RuleBasedInterventionPlanner(include_agent_prompt=False)
        plan = planner.build_plan(
            PredictionRecord(
                student_id="student_0001",
                predicted_grade=8.5,
                risk_factors=["studytime"],
            ),
            [scenario],
        )

        self.assertEqual(plan.risk_band, "medium")
        self.assertEqual(len(plan.recommended_steps), 1)
        self.assertIsNone(plan.llm_prompt)
        self.assertIn("10.5", plan.summary)


if __name__ == "__main__":
    unittest.main()
