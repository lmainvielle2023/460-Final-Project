"""A small intervention catalog keyed by actionable feature changes."""

from __future__ import annotations

from dataclasses import dataclass

from student_success.schemas import InterventionStep, ScenarioChange


@dataclass(frozen=True, slots=True)
class InterventionTemplate:
    title: str
    owner: str
    success_signal: str


_CATALOG = {
    "studytime": InterventionTemplate(
        title="Add structured study support",
        owner="teacher/counselor",
        success_signal="weekly study-time check-ins increase over baseline",
    ),
    "absences": InterventionTemplate(
        title="Launch attendance intervention",
        owner="teacher/administrator",
        success_signal="absences decrease during the next reporting period",
    ),
    "schoolsup": InterventionTemplate(
        title="Enroll in school support services",
        owner="teacher/counselor",
        success_signal="student begins scheduled school support sessions",
    ),
    "famsup": InterventionTemplate(
        title="Coordinate family support plan",
        owner="counselor/guardian",
        success_signal="guardian check-ins are documented and regular",
    ),
    "internet": InterventionTemplate(
        title="Address home access barriers",
        owner="administrator/counselor",
        success_signal="student can access online assignments consistently",
    ),
    "activities": InterventionTemplate(
        title="Increase positive school engagement",
        owner="teacher/advisor",
        success_signal="student participates in at least one structured activity",
    ),
}


def steps_for_changes(changes: list[ScenarioChange]) -> list[InterventionStep]:
    steps: list[InterventionStep] = []
    for change in changes:
        template = _CATALOG.get(change.feature)
        if template is None:
            continue
        steps.append(
            InterventionStep(
                title=template.title,
                owner=template.owner,
                reason=change.rationale,
                success_signal=template.success_signal,
            )
        )
    return steps
