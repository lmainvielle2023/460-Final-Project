"""Shared typed containers used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PredictionRecord:
    """Minimal student prediction bundle used by planners."""

    student_id: str | None
    predicted_grade: float
    risk_factors: list[str] = field(default_factory=list)
    pass_threshold: float = 10.0

    @property
    def risk_band(self) -> str:
        if self.predicted_grade < self.pass_threshold - 2.0:
            return "high"
        if self.predicted_grade < self.pass_threshold:
            return "medium"
        return "low"


@dataclass(slots=True)
class ScenarioChange:
    """One feasible feature change in a simulated intervention path."""

    feature: str
    old_value: object
    new_value: object
    rationale: str


@dataclass(slots=True)
class SimulationResult:
    """A ranked scenario that improves the predicted student outcome."""

    predicted_grade: float
    grade_delta: float
    feasibility_score: float
    changes: list[ScenarioChange]
    meets_target: bool


@dataclass(slots=True)
class InterventionStep:
    """Educator-facing next action."""

    title: str
    owner: str
    reason: str
    success_signal: str


@dataclass(slots=True)
class InterventionPlan:
    """Structured educator-facing plan."""

    student_id: str | None
    risk_band: str
    summary: str
    recommended_steps: list[InterventionStep]
    monitoring_note: str
    llm_prompt: str | None = None
