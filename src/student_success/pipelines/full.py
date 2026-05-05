"""Full experiment, simulation, and report generation pipeline."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from student_success.data.loaders import load_student_performance
from student_success.features.engineering import InteractionSpec, parse_interaction_specs
from student_success.interventions.agent import AgenticInterventionPlanner
from student_success.models.training import (
    FittedModelResult,
    extract_feature_importance,
    fit_and_evaluate_model,
    prepare_model_frame,
)
from student_success.schemas import PredictionRecord
from student_success.settings import ProjectConfig, load_config
from student_success.simulation.recourse import ScenarioSimulator


def ensure_output_dirs(config: ProjectConfig) -> None:
    for raw_path in (
        config.outputs.reports_dir,
        config.outputs.figures_dir,
        config.outputs.models_dir,
    ):
        Path(raw_path).mkdir(parents=True, exist_ok=True)


def _experiment_id(include_prior_grades: bool, use_interactions: bool) -> str:
    prior = "with_prior_grades" if include_prior_grades else "early_warning"
    interactions = "with_interactions" if use_interactions else "baseline_features"
    return f"{prior}__{interactions}"


def _interaction_specs(config: ProjectConfig, use_interactions: bool) -> list[InteractionSpec] | None:
    if not use_interactions:
        return None
    return parse_interaction_specs(config.features.interaction_pairs)


def _result_rows(
    result: FittedModelResult,
    experiment_id: str,
    include_prior_grades: bool,
    use_interactions: bool,
    interpretable_models: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "experiment_id": experiment_id,
        "model": result.model_name,
        "include_prior_grades": include_prior_grades,
        "use_interactions": use_interactions,
        "is_interpretable": result.model_name in interpretable_models,
        "is_exploratory": result.model_name == "mlp",
    }

    test_row = common | {key: value for key, value in result.test_metrics.items() if key != "model"}
    cv_row = common | {key: value for key, value in result.cv_metrics.items() if key != "model"}
    return test_row, cv_row


def run_experiments(config_path: str | Path) -> dict[str, Path]:
    """Run proposal-faithful model comparisons and write report artifacts."""
    config = load_config(config_path)
    ensure_output_dirs(config)
    frame = load_student_performance(
        raw_dir=config.dataset.raw_dir,
        subject=config.dataset.subject,
    )

    test_rows: list[dict[str, Any]] = []
    cv_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    fitted: list[tuple[FittedModelResult, dict[str, Any]]] = []

    for include_prior_grades in (False, True):
        for use_interactions in (False, True):
            experiment_id = _experiment_id(include_prior_grades, use_interactions)
            specs = _interaction_specs(config, use_interactions)

            for model_name in config.modeling.candidate_models:
                result = fit_and_evaluate_model(
                    frame=frame,
                    target=config.dataset.target,
                    model_name=model_name,
                    test_size=config.modeling.test_size,
                    pass_threshold=config.dataset.pass_threshold,
                    random_state=config.project.random_seed,
                    include_prior_grades=include_prior_grades,
                    interaction_specs=specs,
                    cross_validation_folds=config.modeling.cross_validation_folds,
                    max_mlp_iter=config.modeling.max_mlp_iter,
                )
                test_row, cv_row = _result_rows(
                    result=result,
                    experiment_id=experiment_id,
                    include_prior_grades=include_prior_grades,
                    use_interactions=use_interactions,
                    interpretable_models=config.modeling.interpretable_models,
                )
                test_rows.append(test_row)
                cv_rows.append(cv_row)
                fitted.append((result, test_row | cv_row))

                importance = extract_feature_importance(result.pipeline, result.model_name)
                if not importance.empty:
                    importance["experiment_id"] = experiment_id
                    importance["include_prior_grades"] = include_prior_grades
                    importance["use_interactions"] = use_interactions
                    importance_rows.append(importance)

    metrics = pd.DataFrame(test_rows).sort_values(["rmse", "mae"])
    cv_metrics = pd.DataFrame(cv_rows).sort_values(["cv_rmse_mean", "cv_mae_mean"])
    ablation = _build_ablation_summary(cv_metrics)
    feature_importance = (
        pd.concat(importance_rows, ignore_index=True)
        if importance_rows
        else pd.DataFrame(columns=["feature", "importance", "absolute_importance", "model"])
    )

    selected_result, selected_metadata = _select_early_warning_model(
        fitted=fitted,
        config=config,
    )
    _save_model_bundle(
        result=selected_result,
        metadata=selected_metadata,
        config=config,
    )

    reports_dir = Path(config.outputs.reports_dir)
    figures_dir = Path(config.outputs.figures_dir)
    paths = {
        "metrics": reports_dir / "metrics.csv",
        "cross_validation_metrics": reports_dir / "cross_validation_metrics.csv",
        "ablation_summary": reports_dir / "ablation_summary.csv",
        "feature_importance": reports_dir / "feature_importance.csv",
    }
    metrics.to_csv(paths["metrics"], index=False)
    cv_metrics.to_csv(paths["cross_validation_metrics"], index=False)
    ablation.to_csv(paths["ablation_summary"], index=False)
    feature_importance.to_csv(paths["feature_importance"], index=False)

    _plot_model_comparison(cv_metrics, figures_dir / "model_comparison.png")
    _plot_prior_ablation(cv_metrics, figures_dir / "g1_g2_ablation.png")
    _plot_interaction_ablation(cv_metrics, figures_dir / "interaction_ablation.png")
    _plot_top_features(feature_importance, selected_metadata, figures_dir / "top_features.png")
    return paths


def run_simulation(config_path: str | Path) -> dict[str, Path]:
    config = load_config(config_path)
    ensure_output_dirs(config)
    bundle = joblib.load(config.outputs.selected_model_path)
    frame = load_student_performance(
        raw_dir=config.dataset.raw_dir,
        subject=config.dataset.subject,
    )
    metadata = bundle["metadata"]
    pipeline = bundle["pipeline"]
    specs = _interaction_specs(config, bool(metadata["use_interactions"]))

    def score_fn(candidate_frame: pd.DataFrame):
        x, _ = prepare_model_frame(
            frame=candidate_frame.copy(),
            target=config.dataset.target,
            include_prior_grades=bool(metadata["include_prior_grades"]),
            interaction_specs=specs,
        )
        return pipeline.predict(x)

    predictions = score_fn(frame)
    work = frame.copy()
    work["student_id"] = [f"student_{index:04d}" for index in work.index]
    work["predicted_grade"] = predictions
    at_risk = (
        work[work["predicted_grade"] < config.dataset.pass_threshold]
        .sort_values("predicted_grade")
        .head(config.simulation.sample_size)
    )

    simulator = ScenarioSimulator(
        score_fn=score_fn,
        target_grade=config.simulation.target_grade,
        top_k=config.simulation.top_k,
        max_feature_changes=config.simulation.max_feature_changes,
    )
    planner = AgenticInterventionPlanner()

    scenario_rows: list[dict[str, Any]] = []
    plan_blocks: list[str] = []

    for _, student in at_risk.iterrows():
        scenarios = simulator.rank_scenarios(
            student_row=student.drop(labels=["student_id", "predicted_grade"]),
            base_prediction=float(student["predicted_grade"]),
        )
        risk_factors = _risk_factors_from_scenarios(scenarios)
        prediction = PredictionRecord(
            student_id=str(student["student_id"]),
            predicted_grade=float(student["predicted_grade"]),
            risk_factors=risk_factors,
            pass_threshold=config.dataset.pass_threshold,
        )
        plan = planner.build_plan(prediction, scenarios)
        plan_blocks.append(_format_llm_plan_markdown(prediction.student_id, prediction.risk_band, plan))

        if not scenarios:
            scenario_rows.append(
                {
                    "student_id": student["student_id"],
                    "actual_grade": student[config.dataset.target],
                    "predicted_grade": student["predicted_grade"],
                    "risk_band": prediction.risk_band,
                    "scenario_rank": None,
                    "scenario_predicted_grade": None,
                    "grade_delta": None,
                    "meets_target": False,
                    "changes": "",
                }
            )
            continue

        for rank, scenario in enumerate(scenarios, start=1):
            scenario_rows.append(
                {
                    "student_id": student["student_id"],
                    "actual_grade": student[config.dataset.target],
                    "predicted_grade": student["predicted_grade"],
                    "risk_band": prediction.risk_band,
                    "scenario_rank": rank,
                    "scenario_predicted_grade": scenario.predicted_grade,
                    "grade_delta": scenario.grade_delta,
                    "feasibility_score": scenario.feasibility_score,
                    "meets_target": scenario.meets_target,
                    "changes": "; ".join(
                        f"{change.feature}: {change.old_value} -> {change.new_value}"
                        for change in scenario.changes
                    ),
                }
            )

    scenario_summary = pd.DataFrame(scenario_rows)
    reports_dir = Path(config.outputs.reports_dir)
    figures_dir = Path(config.outputs.figures_dir)
    scenario_path = reports_dir / "scenario_summary.csv"
    plans_path = reports_dir / "sample_intervention_plans.md"
    scenario_summary.to_csv(scenario_path, index=False)
    plans_path.write_text("\n\n".join(plan_blocks) + "\n", encoding="utf-8")
    _plot_scenario_lift(scenario_summary, figures_dir / "scenario_lift.png")

    return {
        "scenario_summary": scenario_path,
        "sample_intervention_plans": plans_path,
    }


def build_report(config_path: str | Path) -> Path:
    config = load_config(config_path)
    reports_dir = Path(config.outputs.reports_dir)
    metrics = pd.read_csv(reports_dir / "metrics.csv")
    cv_metrics = pd.read_csv(reports_dir / "cross_validation_metrics.csv")
    ablation = pd.read_csv(reports_dir / "ablation_summary.csv")
    scenarios_path = reports_dir / "scenario_summary.csv"
    scenarios = pd.read_csv(scenarios_path) if scenarios_path.exists() else pd.DataFrame()

    best_cv = cv_metrics.sort_values("cv_rmse_mean").iloc[0]
    early_warning = (
        cv_metrics[
            (~cv_metrics["include_prior_grades"])
            & (cv_metrics["is_interpretable"])
        ]
        .sort_values("cv_rmse_mean")
        .iloc[0]
    )
    best_test = metrics.sort_values("rmse").iloc[0]

    lines = [
        "# Final Results Summary",
        "",
        "## Project Alignment",
        "",
        "This build preserves the original proposal by predicting `G3` on the Portuguese UCI Student Performance dataset with one-hot preprocessing, an 80/20 validation split, 5-fold cross-validation, and a model progression from linear regression to decision trees and MLP. The TA feedback is addressed through interaction features, actionable scenario simulation, cited related work, and rule-based educator intervention plans.",
        "",
        "## Best Models",
        "",
        f"- Best cross-validated model: `{best_cv['model']}` in `{best_cv['experiment_id']}` with RMSE `{best_cv['cv_rmse_mean']:.3f}`.",
        f"- Best 80/20 test model: `{best_test['model']}` in `{best_test['experiment_id']}` with RMSE `{best_test['rmse']:.3f}` and pass/fail accuracy `{best_test['pass_fail_accuracy']:.3f}`.",
        f"- Selected early-warning model: `{early_warning['model']}` in `{early_warning['experiment_id']}` with RMSE `{early_warning['cv_rmse_mean']:.3f}`.",
        "",
        "## Ablation Findings",
        "",
    ]

    for _, row in ablation.iterrows():
        lines.append(
            f"- `{row['comparison']}`: best model `{row['model']}`, CV RMSE `{row['cv_rmse_mean']:.3f}`, MAE `{row['cv_mae_mean']:.3f}`."
        )

    if not scenarios.empty:
        best_scenarios = scenarios.dropna(subset=["grade_delta"])
        students = scenarios["student_id"].nunique()
        avg_lift = float(best_scenarios.groupby("student_id")["grade_delta"].max().mean()) if not best_scenarios.empty else 0.0
        reachable = int(
            best_scenarios.groupby("student_id")["meets_target"].max().sum()
        ) if not best_scenarios.empty else 0
        lines.extend(
            [
                "",
                "## Scenario Simulation",
                "",
                f"- Simulated interventions for `{students}` at-risk students.",
                f"- Average best predicted grade lift: `{avg_lift:.3f}`.",
                f"- Students with at least one scenario reaching the pass threshold: `{reachable}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Generated Artifacts",
            "",
            "- `reports/metrics.csv`",
            "- `reports/cross_validation_metrics.csv`",
            "- `reports/ablation_summary.csv`",
            "- `reports/feature_importance.csv`",
            "- `reports/scenario_summary.csv`",
            "- `reports/sample_intervention_plans.md`",
            "- `reports/figures/`",
        ]
    )

    path = reports_dir / "final_results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_all(config_path: str | Path) -> dict[str, Path]:
    paths = run_experiments(config_path)
    paths.update(run_simulation(config_path))
    paths["final_results_summary"] = build_report(config_path)
    return paths


def _build_ablation_summary(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    labels = [
        ("Early warning, baseline features", False, False),
        ("Early warning, interaction features", False, True),
        ("Prior grades, baseline features", True, False),
        ("Prior grades, interaction features", True, True),
    ]
    rows: list[pd.Series] = []
    for label, include_prior_grades, use_interactions in labels:
        subset = cv_metrics[
            (cv_metrics["include_prior_grades"] == include_prior_grades)
            & (cv_metrics["use_interactions"] == use_interactions)
        ]
        if subset.empty:
            continue
        row = subset.sort_values("cv_rmse_mean").iloc[0].copy()
        row["comparison"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def _select_early_warning_model(
    fitted: list[tuple[FittedModelResult, dict[str, Any]]],
    config: ProjectConfig,
) -> tuple[FittedModelResult, dict[str, Any]]:
    candidates = [
        (result, metadata)
        for result, metadata in fitted
        if not metadata["include_prior_grades"]
        and metadata["model"] in config.modeling.interpretable_models
    ]
    if not candidates:
        raise RuntimeError("No interpretable early-warning model was fitted.")

    result, metadata = min(candidates, key=lambda item: item[1]["cv_rmse_mean"])
    metadata = dict(metadata)
    metadata["selection_reason"] = "Best CV RMSE among interpretable models excluding G1/G2."
    return result, metadata


def _save_model_bundle(
    result: FittedModelResult,
    metadata: dict[str, Any],
    config: ProjectConfig,
) -> None:
    bundle = {
        "pipeline": result.pipeline,
        "metadata": {
            "model": result.model_name,
            "target": config.dataset.target,
            "pass_threshold": config.dataset.pass_threshold,
            "interaction_pairs": config.features.interaction_pairs,
            **metadata,
        },
        "config": asdict(config),
    }
    joblib.dump(bundle, config.outputs.selected_model_path)
    metadata_path = Path(config.outputs.models_dir) / "selected_early_warning_model_metadata.json"
    metadata_path.write_text(json.dumps(bundle["metadata"], indent=2), encoding="utf-8")


def _risk_factors_from_scenarios(scenarios) -> list[str]:
    factors: list[str] = []
    if not scenarios:
        return ["low predicted grade"]
    for change in scenarios[0].changes:
        if change.feature not in factors:
            factors.append(change.feature)
    return factors or ["low predicted grade"]


def _format_llm_plan_markdown(student_id: str, risk_band: str, plan) -> str:
    lines = [
        f"## {student_id}",
        "",
        f"Risk band: **{risk_band}**",
        "",
        plan.summary,
        "",
        "Recommended steps:",
    ]
    for step in plan.recommended_steps:
        lines.append(
            f"- **{step.title}** ({step.owner}): {step.description}"
        )
    lines.extend(["", f"Monitoring: {plan.monitoring_note}"])
    return "\n".join(lines)


def _plot_model_comparison(cv_metrics: pd.DataFrame, path: Path) -> None:
    summary = cv_metrics.groupby("model", as_index=False)["cv_rmse_mean"].min()
    summary = summary.sort_values("cv_rmse_mean")
    _bar_plot(
        labels=summary["model"].tolist(),
        values=summary["cv_rmse_mean"].tolist(),
        title="Best 5-Fold CV RMSE By Model",
        ylabel="RMSE",
        path=path,
    )


def _plot_prior_ablation(cv_metrics: pd.DataFrame, path: Path) -> None:
    summary = (
        cv_metrics.groupby("include_prior_grades", as_index=False)["cv_rmse_mean"]
        .min()
        .sort_values("include_prior_grades")
    )
    labels = ["without G1/G2" if not value else "with G1/G2" for value in summary["include_prior_grades"]]
    _bar_plot(
        labels=labels,
        values=summary["cv_rmse_mean"].tolist(),
        title="Effect Of Including Prior Grades",
        ylabel="Best CV RMSE",
        path=path,
    )


def _plot_interaction_ablation(cv_metrics: pd.DataFrame, path: Path) -> None:
    early = cv_metrics[~cv_metrics["include_prior_grades"]]
    summary = (
        early.groupby("use_interactions", as_index=False)["cv_rmse_mean"]
        .min()
        .sort_values("use_interactions")
    )
    labels = ["baseline" if not value else "interactions" for value in summary["use_interactions"]]
    _bar_plot(
        labels=labels,
        values=summary["cv_rmse_mean"].tolist(),
        title="Effect Of Interaction Features",
        ylabel="Best Early-Warning CV RMSE",
        path=path,
    )


def _plot_top_features(
    feature_importance: pd.DataFrame,
    selected_metadata: dict[str, Any],
    path: Path,
) -> None:
    subset = feature_importance[
        (feature_importance["model"] == selected_metadata["model"])
        & (feature_importance["experiment_id"] == selected_metadata["experiment_id"])
    ].head(12)
    if subset.empty:
        _placeholder_png(path, "Top Features", "No feature importance available.")
        return

    ordered = subset.sort_values("absolute_importance", ascending=True)
    _bar_plot(
        labels=ordered["feature"].tolist(),
        values=ordered["importance"].tolist(),
        title="Top Features In Selected Early-Warning Model",
        ylabel="Importance",
        path=path,
    )


def _plot_scenario_lift(scenario_summary: pd.DataFrame, path: Path) -> None:
    if scenario_summary.empty or "grade_delta" not in scenario_summary:
        _placeholder_png(path, "Scenario Lift", "No scenario data available.")
        return
    best = (
        scenario_summary.dropna(subset=["grade_delta"])
        .sort_values("grade_delta", ascending=False)
        .groupby("student_id", as_index=False)
        .first()
        .sort_values("grade_delta", ascending=False)
    )
    if best.empty:
        _placeholder_png(path, "Scenario Lift", "No positive scenario lift available.")
        return
    _bar_plot(
        labels=best["student_id"].tolist(),
        values=best["grade_delta"].tolist(),
        title="Best Simulated Grade Lift For At-Risk Students",
        ylabel="Predicted grade lift",
        path=path,
    )


def _bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    width = 1100
    height = max(420, 115 + 44 * len(labels))
    image = Image.new("RGB", (width, height), "#f8faf8")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=20)

    draw.text((32, 24), title, fill="#1f2d2e", font=title_font)
    draw.text((32, 54), ylabel, fill="#4b5d5f", font=font)

    clean_values = [float(value) for value in values]
    min_value = min(clean_values + [0.0])
    max_value = max(clean_values + [0.0])
    value_span = max(max_value - min_value, 1e-9)

    label_width = 285
    chart_left = 340
    chart_right = width - 70
    chart_width = chart_right - chart_left
    zero_x = chart_left + int((0.0 - min_value) / value_span * chart_width)
    top = 92
    row_height = 42

    draw.line((chart_left, top - 12, chart_right, top - 12), fill="#ced8d9", width=1)
    draw.line((zero_x, top - 18, zero_x, height - 36), fill="#7c9194", width=1)

    for index, (label, value) in enumerate(zip(labels, clean_values, strict=True)):
        y = top + index * row_height
        label_text = str(label)
        if len(label_text) > 42:
            label_text = f"{label_text[:39]}..."
        draw.text((32, y + 5), label_text, fill="#243638", font=font)

        value_x = chart_left + int((value - min_value) / value_span * chart_width)
        x0, x1 = sorted((zero_x, value_x))
        draw.rounded_rectangle(
            (x0, y, max(x1, x0 + 2), y + 22),
            radius=3,
            fill="#2f6f73" if value >= 0 else "#9d4f3d",
        )
        draw.text((max(x0, x1) + 8, y + 5), f"{value:.3f}", fill="#243638", font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _placeholder_png(path: Path, title: str, message: str) -> None:
    image = Image.new("RGB", (900, 360), "#f8faf8")
    draw = ImageDraw.Draw(image)
    draw.text((32, 32), title, fill="#1f2d2e", font=ImageFont.load_default(size=20))
    draw.text((32, 72), message, fill="#4b5d5f", font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
