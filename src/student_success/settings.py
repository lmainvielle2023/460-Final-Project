"""Configuration loading for the project scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(slots=True)
class ProjectMetaConfig:
    name: str
    random_seed: int


@dataclass(slots=True)
class DatasetConfig:
    raw_dir: str
    subject: str
    target: str
    include_prior_grades: bool
    pass_threshold: float


@dataclass(slots=True)
class FeatureConfig:
    generate_interactions: bool
    interaction_pairs: list[str]


@dataclass(slots=True)
class ModelingConfig:
    candidate_models: list[str]
    interpretable_models: list[str]
    test_size: float
    cross_validation_folds: int
    max_mlp_iter: int


@dataclass(slots=True)
class SimulationConfig:
    target_grade: float
    top_k: int
    max_feature_changes: int
    sample_size: int


@dataclass(slots=True)
class InterventionConfig:
    planner_mode: str
    include_agent_prompt: bool


@dataclass(slots=True)
class OutputConfig:
    reports_dir: str
    figures_dir: str
    models_dir: str
    selected_model_path: str


@dataclass(slots=True)
class ProjectConfig:
    project: ProjectMetaConfig
    dataset: DatasetConfig
    features: FeatureConfig
    modeling: ModelingConfig
    simulation: SimulationConfig
    intervention: InterventionConfig
    outputs: OutputConfig


def load_config(path: str | Path) -> ProjectConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

    return ProjectConfig(
        project=ProjectMetaConfig(**raw["project"]),
        dataset=DatasetConfig(**raw["dataset"]),
        features=FeatureConfig(**raw["features"]),
        modeling=ModelingConfig(**raw["modeling"]),
        simulation=SimulationConfig(**raw["simulation"]),
        intervention=InterventionConfig(**raw["intervention"]),
        outputs=OutputConfig(**raw["outputs"]),
    )
