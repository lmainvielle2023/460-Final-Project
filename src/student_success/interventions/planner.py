"""Rule-based planning plus prompt scaffolding for a later agentic layer."""

from __future__ import annotations

from student_success.interventions.catalog import steps_for_changes
from student_success.schemas import InterventionPlan, PredictionRecord, SimulationResult


def build_agent_prompt(
    prediction: PredictionRecord,
    scenarios: list[SimulationResult],
) -> str:
    top_scenarios = scenarios[:3]
    scenario_lines = []
    for index, scenario in enumerate(top_scenarios, start=1):
        change_text = ", ".join(
            f"{change.feature}: {change.old_value} -> {change.new_value}"
            for change in scenario.changes
        )
        scenario_lines.append(
            f"{index}. predicted_grade={scenario.predicted_grade:.2f}; "
            f"delta={scenario.grade_delta:.2f}; changes={change_text}"
        )

    risk_text = ", ".join(prediction.risk_factors) if prediction.risk_factors else "not provided"
    joined_scenarios = "\n".join(scenario_lines) if scenario_lines else "No positive scenarios found."
    return (
        "You are generating a concise educator-facing intervention plan.\n"
        f"Predicted grade: {prediction.predicted_grade:.2f}\n"
        f"Risk band: {prediction.risk_band}\n"
        f"Top risk factors: {risk_text}\n"
        "Best feasible scenarios:\n"
        f"{joined_scenarios}\n"
        "Return: summary, 2-4 actions, and a short monitoring plan."
    )

"""Old class, replaced with the AI agent planner"""
class RuleBasedInterventionPlanner:
    """Turn predictions and scenario simulations into a short support plan."""

    def __init__(self, include_agent_prompt: bool = True) -> None:
        self.include_agent_prompt = include_agent_prompt

    def build_plan(
        self,
        prediction: PredictionRecord,
        scenarios: list[SimulationResult],
    ) -> InterventionPlan:
        best_scenario = scenarios[0] if scenarios else None
        steps = steps_for_changes(best_scenario.changes if best_scenario else [])

        if best_scenario is None:
            summary = (
                f"Predicted grade is {prediction.predicted_grade:.1f} "
                f"with {prediction.risk_band} risk and no positive simulated "
                "scenarios under current constraints."
            )
            monitoring_note = "Review feature constraints and gather more recent student data."
        else:
            risk_text = ", ".join(prediction.risk_factors) if prediction.risk_factors else "general academic risk"
            summary = (
                f"Predicted grade is {prediction.predicted_grade:.1f} "
                f"({prediction.risk_band} risk). The best simulated path raises "
                f"the estimate to {best_scenario.predicted_grade:.1f} by targeting "
                f"{risk_text}."
            )
            monitoring_note = (
                "Re-evaluate after the next grading or attendance checkpoint and "
                "track whether the targeted features move in the desired direction."
            )

        prompt = build_agent_prompt(prediction, scenarios) if self.include_agent_prompt else None
        return InterventionPlan(
            student_id=prediction.student_id,
            risk_band=prediction.risk_band,
            summary=summary,
            recommended_steps=steps,
            monitoring_note=monitoring_note,
            llm_prompt=prompt,
        )
