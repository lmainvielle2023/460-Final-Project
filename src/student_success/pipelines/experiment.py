"""High-level experiment runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from student_success.data.loaders import load_student_performance
from student_success.features.engineering import parse_interaction_specs
from student_success.models.training import score_candidate_models
from student_success.settings import load_config


def run_experiment(config_path: str | Path) -> pd.DataFrame:
    config = load_config(config_path)
    frame = load_student_performance(
        raw_dir=config.dataset.raw_dir,
        subject=config.dataset.subject,
    )

    interaction_specs = None
    if config.features.generate_interactions:
        interaction_specs = parse_interaction_specs(config.features.interaction_pairs)

    return score_candidate_models(
        frame=frame,
        target=config.dataset.target,
        candidate_models=config.modeling.candidate_models,
        test_size=config.modeling.test_size,
        pass_threshold=config.dataset.pass_threshold,
        random_state=config.project.random_seed,
        include_prior_grades=config.dataset.include_prior_grades,
        interaction_specs=interaction_specs,
        cross_validation_folds=config.modeling.cross_validation_folds,
        max_mlp_iter=config.modeling.max_mlp_iter,
    )
