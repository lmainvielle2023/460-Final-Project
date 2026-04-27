# Final Results Summary

## Project Alignment

This build preserves the original proposal by predicting `G3` on the Portuguese UCI Student Performance dataset with one-hot preprocessing, an 80/20 validation split, 5-fold cross-validation, and a model progression from linear regression to decision trees and MLP. The TA feedback is addressed through interaction features, actionable scenario simulation, cited related work, and rule-based educator intervention plans.

## Best Models

- Best cross-validated model: `lasso` in `with_prior_grades__with_interactions` with RMSE `1.273`.
- Best 80/20 test model: `lasso` in `with_prior_grades__with_interactions` with RMSE `1.172` and pass/fail accuracy `0.900`.
- Selected early-warning model: `lasso` in `early_warning__with_interactions` with RMSE `2.725`.

## Ablation Findings

- `Early warning, baseline features`: best model `lasso`, CV RMSE `2.736`, MAE `1.989`.
- `Early warning, interaction features`: best model `lasso`, CV RMSE `2.725`, MAE `1.987`.
- `Prior grades, baseline features`: best model `lasso`, CV RMSE `1.277`, MAE `0.825`.
- `Prior grades, interaction features`: best model `lasso`, CV RMSE `1.273`, MAE `0.821`.

## Scenario Simulation

- Simulated interventions for `12` at-risk students.
- Average best predicted grade lift: `1.101`.
- Students with at least one scenario reaching the pass threshold: `0`.

## Generated Artifacts

- `reports/metrics.csv`
- `reports/cross_validation_metrics.csv`
- `reports/ablation_summary.csv`
- `reports/feature_importance.csv`
- `reports/scenario_summary.csv`
- `reports/sample_intervention_plans.md`
- `reports/figures/`
