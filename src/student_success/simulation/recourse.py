"""Simple scenario simulator for actionable feature changes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations, product

import pandas as pd

from student_success.schemas import ScenarioChange, SimulationResult


@dataclass(frozen=True, slots=True)
class ActionableFeatureSpec:
    feature: str
    candidate_values: tuple[object, ...]
    cost: float = 1.0
    rationale: str = ""

    def valid_values_for(self, current_value: object) -> list[object]:
        return [value for value in self.candidate_values if value != current_value]


def default_action_space() -> dict[str, ActionableFeatureSpec]:
    return {
        "studytime": ActionableFeatureSpec(
            feature="studytime",
            candidate_values=(2, 3, 4),
            cost=1.0,
            rationale="Increase structured study support or tutoring time.",
        ),
        "absences": ActionableFeatureSpec(
            feature="absences",
            candidate_values=(0, 1, 2, 4),
            cost=1.2,
            rationale="Reduce missed instruction through attendance support.",
        ),
        "schoolsup": ActionableFeatureSpec(
            feature="schoolsup",
            candidate_values=("yes",),
            cost=0.8,
            rationale="Enroll the student in school support programs.",
        ),
        "famsup": ActionableFeatureSpec(
            feature="famsup",
            candidate_values=("yes",),
            cost=0.9,
            rationale="Coordinate additional home or guardian support.",
        ),
        "internet": ActionableFeatureSpec(
            feature="internet",
            candidate_values=("yes",),
            cost=1.1,
            rationale="Improve access to homework and learning resources.",
        ),
        "activities": ActionableFeatureSpec(
            feature="activities",
            candidate_values=("yes",),
            cost=0.7,
            rationale="Increase structured school engagement.",
        ),
    }


class ScenarioSimulator:
    """Rank feasible feature changes by predicted grade improvement."""

    def __init__(
        self,
        score_fn: Callable[[pd.DataFrame], Sequence[float]],
        action_space: dict[str, ActionableFeatureSpec] | None = None,
        target_grade: float = 10.0,
        top_k: int = 5,
        max_feature_changes: int = 2,
    ) -> None:
        self.score_fn = score_fn
        self.action_space = action_space or default_action_space()
        self.target_grade = target_grade
        self.top_k = top_k
        self.max_feature_changes = max_feature_changes

    def rank_scenarios(
        self,
        student_row: pd.Series,
        base_prediction: float | None = None,
    ) -> list[SimulationResult]:
        student_frame = pd.DataFrame([student_row])
        if base_prediction is None:
            base_prediction = float(self.score_fn(student_frame)[0])

        scenarios: list[SimulationResult] = []
        specs = list(self.action_space.values())

        for width in range(1, self.max_feature_changes + 1):
            for feature_group in combinations(specs, width):
                candidate_lists = [
                    spec.valid_values_for(student_row.get(spec.feature)) for spec in feature_group
                ]
                if any(not values for values in candidate_lists):
                    continue

                for values in product(*candidate_lists):
                    candidate = student_row.copy()
                    changes: list[ScenarioChange] = []
                    total_cost = 0.0

                    for spec, new_value in zip(feature_group, values, strict=True):
                        old_value = candidate.get(spec.feature)
                        candidate[spec.feature] = new_value
                        total_cost += spec.cost
                        changes.append(
                            ScenarioChange(
                                feature=spec.feature,
                                old_value=old_value,
                                new_value=new_value,
                                rationale=spec.rationale,
                            )
                        )

                    predicted_grade = float(self.score_fn(pd.DataFrame([candidate]))[0])
                    grade_delta = predicted_grade - base_prediction
                    if grade_delta <= 0:
                        continue

                    feasibility_score = grade_delta / total_cost if total_cost else grade_delta
                    scenarios.append(
                        SimulationResult(
                            predicted_grade=predicted_grade,
                            grade_delta=grade_delta,
                            feasibility_score=feasibility_score,
                            changes=changes,
                            meets_target=predicted_grade >= self.target_grade,
                        )
                    )

        scenarios.sort(
            key=lambda item: (
                item.meets_target,
                item.grade_delta,
                item.feasibility_score,
            ),
            reverse=True,
        )
        return scenarios[: self.top_k]
