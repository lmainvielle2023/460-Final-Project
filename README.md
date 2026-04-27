# Student Success Intervention Pipeline

This project goes beyond merely predicting student performance. Instead of stopping at predicting a final grade (`G3`), it focuses on answering a more actionable question:

**Which realistic changes could help an at-risk student, and how should an educator intervene?**

## What This Project Accomplishes

This project provides an end-to-end machine learning pipeline that acts as an **Early-Warning and Intervention System**. It accomplishes the following:
1. **Prediction & Early Warning**: Identifies students at risk of failing by utilizing interpretable models built on demographic, social, and behavioral features. It includes ablation studies comparing models with and without prior grades (`G1`/`G2`).
2. **Feature Engineering**: Incorporates engineered interaction features (e.g., `studytime x absences`, `studytime x failures`) to capture nuanced relationships and improve model explainability.
3. **Actionable Scenario Simulation**: Calculates how feasible changes to actionable variables (e.g., increasing study time, reducing absences, accessing school support) shift a student's predicted outcome.
4. **Intervention Planning**: Translates the top simulated scenarios and risk factors into concise, educator-facing support plans.

## What Has Been Done (Current Status)

All initial project goals and immediate priorities have been fully executed:
- The UCI Student Performance dataset has been successfully integrated.
- Baseline and ablation experiments have been run (comparing models with/without prior grades and with/without interaction features).
- The scenario simulator has generated top-k feasible interventions for at-risk students, measuring the average predicted lift.
- The rule-based planner has synthesized these results into concrete support plans for educators.
- **Results and Artifacts**: All final metrics, feature importances, ablation summaries, and sample intervention plans have been generated and reside in the `reports/` directory.

## File Structure & Where Things Live

- `configs/`: Contains experiment settings (`default.toml`). Modify this to adjust thresholds, simulation parameters, and model selections.
- `data/`: 
  - `raw/`: Where the raw UCI datasets (e.g., `student-por.csv`) live.
  - `interim/` & `processed/`: Intermediate and final dataset locations.
- `docs/`: Includes the revised project outline and deeper documentation on project framing.
- `reports/`: **This is where all generated outputs live.**
  - `figures/`: Plots showing model comparisons, ablation impacts, and feature importances.
  - `final_results_summary.md`: The main summary of the model experiments.
  - `sample_intervention_plans.md`: Generated support plans for educators.
  - `metrics.csv`, `cross_validation_metrics.csv`, `ablation_summary.csv`, `scenario_summary.csv`: Raw tabular results.
- `models/`: Saved model bundles and metadata (e.g., the selected early warning model).
- `src/student_success/`: The core source code for the pipeline.
  - `cli.py`: The command-line interface.
  - `data/`: Data loading and preprocessing.
  - `features/`: Feature engineering and interaction specifications.
  - `models/`: Model training and evaluation logic.
  - `simulation/`: The counterfactual scenario simulator.
  - `interventions/`: The rule-based intervention planner.
  - `pipelines/`: The full execution flow connecting all steps.
- `tests/`: Unit tests ensuring the pipeline's logic remains intact.

## How It Works (Running the Pipeline)

The project is built as a Python package. The main entry point is the `student-success` CLI.

1. **Install Dependencies**: `pip install -e .`
2. **Verify Setup**: `student-success validate-layout`
3. **Run Everything**: `student-success run-all`
   - This command runs model training, cross-validation, ablation experiments, scenario simulation, and report building sequentially. It places all final artifacts in the `reports/` folder.

You can also run specific segments of the pipeline, such as `train`, `run-experiments`, or `simulate`. Use `student-success --help` for details.

## Related Work & Existing Approaches

This project builds upon established literature by shifting from static prediction to actionable recourse:

- **Student Performance Prediction**: While Cortez and Silva (2008) showed student performance could be modeled using demographic and school-related variables, relying heavily on `G1` and `G2` reduces the usefulness of *early* intervention. Namoun and Alshanqiti (2021) highlight that explanatory analytics in this field remain limited, supporting our focus on interpretability over black-box models.
- **Interpretability and Early Warning**: Tools like SHAP (Lundberg and Lee, 2017) provide necessary local explanations. As Xia and Qi (2023) demonstrate, educational prediction becomes truly valuable only when tied to understandable recommendations.
- **Actionable Recommendations and Recourse**: Wachter et al. (2018) and Ustun et al. (2019) formalize actionable recourse—focusing recommendations on variables a person or institution can realistically change. We separate immutable attributes from intervention-relevant features (e.g., support, attendance) to simulate realistic uplift.
