# Student Success Intervention Pipeline

This project goes beyond merely predicting student performance. Instead of stopping at predicting a final grade (`G3`), it focuses on answering a more actionable question:

**Which realistic changes could help an at-risk student, and how should an educator intervene?**

## What This Project Accomplishes

This project provides an end-to-end machine learning pipeline that acts as an **Early-Warning and Intervention System**. It accomplishes the following:
1. **Prediction & Early Warning**: Identifies students at risk of failing by utilizing interpretable models built on demographic, social, and behavioral features. It includes ablation studies comparing models with and without prior grades (`G1`/`G2`).
2. **Feature Engineering**: Incorporates engineered interaction features (e.g., `studytime x absences`, `studytime x failures`) to capture nuanced relationships and improve model explainability.
3. **Actionable Scenario Simulation**: Calculates how feasible changes to actionable variables (e.g., increasing study time, reducing absences, accessing school support) shift a student's predicted outcome.
4. **Agentic Intervention Planning**: Uses an autonomous LangChain tool-calling agent to iteratively simulate combinations of changes and synthesizes the top paths into concise, educator-facing support plans.

## What Has Been Done (Current Status)

All initial project goals and immediate priorities have been fully executed:
- The UCI Student Performance dataset has been successfully integrated.
- Baseline and ablation experiments have been run (comparing models with/without prior grades and with/without interaction features).
- The LangChain agent autonomously generated feasible interventions for at-risk students, interacting with the trained prediction model.
- **Results and Artifacts**: All final metrics, feature importances, ablation summaries, and agent-generated intervention plans have been generated and reside in the `reports/` directory.

## File Structure & Where Things Live

- `configs/`: Contains experiment settings (`default.toml`). Modify this to adjust thresholds, simulation parameters, and model selections.
- `data/`: Where the raw UCI dataset (`student-por.csv`) lives.
- `reports/`: **This is where all generated outputs live.**
  - `figures/`: Plots showing model comparisons, ablation impacts, and feature importances.
  - `final_results_summary.md`: The main summary of the model experiments.
  - `sample_intervention_plans.md`: Generated support plans for educators produced by the LangChain agent.
  - `metrics.csv`, `cross_validation_metrics.csv`, `ablation_summary.csv`, `scenario_summary.csv`: Raw tabular results.
- `models/`: Saved model bundles and metadata (e.g., the selected early warning model).
- `src/student_success/`: The core source code for the pipeline.
  - `cli.py`: The command-line interface.
  - `data/`: Data loading and preprocessing.
  - `features/`: Feature engineering and interaction specifications.
  - `models/`: Model training and evaluation logic.
  - `interventions/`: The LangChain agent and planning logic.
  - `pipelines/`: The full execution flow connecting all steps.
- `tests/`: Unit tests ensuring the pipeline's logic remains intact.

## How to Run the Project (Step-by-Step)

The project is built as a Python package. The main entry point is the `student-success` CLI.

1. **Install Dependencies**: First, install the required packages using `pip`.
   ```bash
   pip install -e .
   ```

2. **Verify Setup**: Ensure the raw data (e.g. `student-por.csv`) is present in `data/`.
   ```bash
   student-success validate-layout
   ```

3. **Configure API Key**: The intervention planner uses a LangChain agent powered by Google's Gemini. You must export your API key before running simulations.
   ```bash
   export GEMINI_API_KEY="your_actual_api_key_here"
   ```
   *(Note: You can also use `GOOGLE_API_KEY` depending on your environment).*

4. **Run the Full Pipeline**: Execute the entire pipeline to train models, run ablation studies, and deploy the agent to generate intervention plans.
   ```bash
   student-success run-all
   ```
   This command runs everything sequentially. Once finished, all final artifacts will be placed in the `reports/` folder.

If you just want to run specific segments of the pipeline, such as `train`, `run-experiments`, or `simulate`, you can use those subcommands instead. Run `student-success --help` for details.
