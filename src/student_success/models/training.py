"""Baseline training pipeline for student outcome prediction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from student_success.dependencies import require_scikit_learn
from student_success.features.engineering import (
    InteractionSpec,
    build_interaction_features,
)
from student_success.models.evaluation import compute_regression_metrics


@dataclass(slots=True)
class FittedModelResult:
    model_name: str
    pipeline: object
    test_metrics: dict[str, float | str | bool]
    cv_metrics: dict[str, float | str | bool]
    x_test: pd.DataFrame
    y_test: pd.Series
    test_predictions: np.ndarray


def prepare_model_frame(
    frame: pd.DataFrame,
    target: str = "G3",
    include_prior_grades: bool = False,
    interaction_specs: Sequence[InteractionSpec] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    modeling_frame = frame.copy()
    if interaction_specs:
        modeling_frame = build_interaction_features(modeling_frame, interaction_specs)

    if not include_prior_grades:
        modeling_frame = modeling_frame.drop(
            columns=[column for column in ("G1", "G2") if column in modeling_frame.columns]
        )

    y = modeling_frame[target].copy() if target in modeling_frame.columns else None
    x = modeling_frame.drop(columns=[target]) if target in modeling_frame.columns else modeling_frame
    return x, y


def _build_preprocessor(x: pd.DataFrame):
    require_scikit_learn()

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def _build_candidate_models(random_state: int, max_mlp_iter: int = 3000) -> dict[str, object]:
    require_scikit_learn()

    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor

    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, max_iter=5000),
        "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            alpha=0.001,
            early_stopping=True,
            max_iter=max_mlp_iter,
            random_state=random_state,
        ),
    }


def build_model_pipeline(
    x: pd.DataFrame,
    model_name: str,
    random_state: int,
    max_mlp_iter: int = 3000,
):
    require_scikit_learn()

    from sklearn.pipeline import Pipeline

    available_models = _build_candidate_models(
        random_state=random_state,
        max_mlp_iter=max_mlp_iter,
    )
    if model_name not in available_models:
        raise ValueError(f"Unsupported model '{model_name}'. Expected one of {sorted(available_models)}.")

    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(x)),
            ("model", available_models[model_name]),
        ]
    )


def _cross_validate_pipeline(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    folds: int,
    pass_threshold: float,
    random_state: int,
    max_mlp_iter: int,
) -> dict[str, float | str]:
    require_scikit_learn()

    from sklearn.base import clone
    from sklearn.model_selection import KFold

    pipeline = build_model_pipeline(
        x=x,
        model_name=model_name,
        random_state=random_state,
        max_mlp_iter=max_mlp_iter,
    )
    splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_rows: list[dict[str, float]] = []

    for train_index, test_index in splitter.split(x):
        fold_pipeline = clone(pipeline)
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        fold_pipeline.fit(x_train, y_train)
        predictions = fold_pipeline.predict(x_test)
        fold_rows.append(
            compute_regression_metrics(
                y_true=y_test.to_numpy(),
                y_pred=predictions,
                pass_threshold=pass_threshold,
            ).as_dict()
        )

    metrics = pd.DataFrame(fold_rows)
    row: dict[str, float | str] = {"model": model_name}
    for column in metrics.columns:
        row[f"cv_{column}_mean"] = float(metrics[column].mean())
        row[f"cv_{column}_std"] = float(metrics[column].std(ddof=0))
    return row


def fit_and_evaluate_model(
    frame: pd.DataFrame,
    target: str,
    model_name: str,
    test_size: float,
    pass_threshold: float,
    random_state: int,
    include_prior_grades: bool,
    interaction_specs: Sequence[InteractionSpec] | None = None,
    cross_validation_folds: int = 5,
    max_mlp_iter: int = 3000,
) -> FittedModelResult:
    require_scikit_learn()

    from sklearn.model_selection import train_test_split

    x, y = prepare_model_frame(
        frame=frame,
        target=target,
        include_prior_grades=include_prior_grades,
        interaction_specs=interaction_specs,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_model_pipeline(
        x=x,
        model_name=model_name,
        random_state=random_state,
        max_mlp_iter=max_mlp_iter,
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    test_metrics = compute_regression_metrics(
        y_true=y_test.to_numpy(),
        y_pred=predictions,
        pass_threshold=pass_threshold,
    ).as_dict()
    test_metrics["model"] = model_name

    cv_metrics = _cross_validate_pipeline(
        x=x,
        y=y,
        model_name=model_name,
        folds=cross_validation_folds,
        pass_threshold=pass_threshold,
        random_state=random_state,
        max_mlp_iter=max_mlp_iter,
    )

    return FittedModelResult(
        model_name=model_name,
        pipeline=pipeline,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
        x_test=x_test,
        y_test=y_test,
        test_predictions=predictions,
    )


def score_candidate_models(
    frame: pd.DataFrame,
    target: str,
    candidate_models: Sequence[str],
    test_size: float,
    pass_threshold: float,
    random_state: int,
    include_prior_grades: bool,
    interaction_specs: Sequence[InteractionSpec] | None = None,
    cross_validation_folds: int = 5,
    max_mlp_iter: int = 3000,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for model_name in candidate_models:
        result = fit_and_evaluate_model(
            frame=frame,
            target=target,
            model_name=model_name,
            test_size=test_size,
            pass_threshold=pass_threshold,
            random_state=random_state,
            include_prior_grades=include_prior_grades,
            interaction_specs=interaction_specs,
            cross_validation_folds=cross_validation_folds,
            max_mlp_iter=max_mlp_iter,
        )
        rows.append(result.test_metrics)

    return pd.DataFrame(rows).sort_values(by=["rmse", "mae"], ascending=[True, True])


def get_transformed_feature_names(pipeline: object) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    return [str(name).replace("numeric__", "").replace("categorical__", "") for name in preprocessor.get_feature_names_out()]


def extract_feature_importance(pipeline: object, model_name: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(pipeline)

    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).reshape(-1)
    else:
        return pd.DataFrame(columns=["feature", "importance", "absolute_importance", "model"])

    rows = pd.DataFrame(
        {
            "feature": feature_names[: len(values)],
            "importance": values,
        }
    )
    rows["absolute_importance"] = rows["importance"].abs()
    rows["model"] = model_name
    return rows.sort_values("absolute_importance", ascending=False).reset_index(drop=True)
