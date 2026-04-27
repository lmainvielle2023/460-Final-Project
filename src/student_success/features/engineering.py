"""Interaction features chosen for interpretability and actionability."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype


_BINARY_MAP = {
    "yes": 1.0,
    "no": 0.0,
    "M": 1.0,
    "F": 0.0,
    "U": 1.0,
    "R": 0.0,
    "T": 1.0,
    "A": 0.0,
    "GP": 1.0,
    "MS": 0.0,
    "GT3": 1.0,
    "LE3": 0.0,
}


@dataclass(frozen=True, slots=True)
class InteractionSpec:
    left: str
    right: str

    @property
    def column_name(self) -> str:
        return f"{self.left}_x_{self.right}"


def _coerce_for_interaction(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        return series.astype(float)

    mapped = series.astype(str).map(_BINARY_MAP)
    if mapped.notna().all():
        return mapped.astype(float)

    categories = {value: index for index, value in enumerate(sorted(series.astype(str).unique()))}
    return series.astype(str).map(categories).astype(float)


def parse_interaction_specs(tokens: Sequence[str]) -> list[InteractionSpec]:
    specs: list[InteractionSpec] = []
    for token in tokens:
        if "*" not in token:
            raise ValueError(f"Invalid interaction pair '{token}'. Expected 'left*right'.")
        left, right = token.split("*", maxsplit=1)
        specs.append(InteractionSpec(left=left.strip(), right=right.strip()))
    return specs


def default_interaction_specs(include_prior_grades: bool = False) -> list[InteractionSpec]:
    specs = [
        InteractionSpec("studytime", "absences"),
        InteractionSpec("studytime", "failures"),
        InteractionSpec("schoolsup", "failures"),
        InteractionSpec("internet", "studytime"),
        InteractionSpec("Medu", "Fedu"),
    ]
    if include_prior_grades:
        specs.append(InteractionSpec("G1", "G2"))
    return specs


def build_interaction_features(
    frame: pd.DataFrame,
    specs: Sequence[InteractionSpec],
) -> pd.DataFrame:
    enriched = frame.copy()
    for spec in specs:
        left = _coerce_for_interaction(enriched[spec.left])
        right = _coerce_for_interaction(enriched[spec.right])
        enriched[spec.column_name] = left * right
    return enriched
