# Student Success Intervention Pipeline

This repository is a project scaffold for a revised version of your proposal. The core change is that the project no longer stops at predicting `G3`; it is structured to answer the more useful question:

**Which realistic changes could help a student, and how should an educator intervene?**

## Revised Project Frame

The TA feedback maps to four concrete workstreams:

1. **Prediction**: build baseline and stronger interpretable models for `G3` and optional pass/fail risk.
2. **Feature engineering**: add interaction features that capture relationships such as attendance-study balance and support-history effects.
3. **Scenario simulation**: estimate how changing actionable features could shift a student’s predicted outcome.
4. **Intervention planning**: translate predictions and simulated improvements into educator-facing action plans.

## Repository Layout

- `configs/`: experiment settings
- `docs/`: revised outline and related-work notes
- `refs/`: seed bibliography for the report
- `data/`: raw, interim, and processed dataset locations
- `src/student_success/`: project code
- `tests/`: lightweight unit tests for scaffolding logic

## Starter Workflow

1. Install dependencies: `pip install -e .`
2. Put the UCI CSV files in `data/raw/`
3. Review `docs/revised_project_outline.md`
4. Start with:
   - `student-success validate-layout`
   - `student-success print-config`

## Immediate Priorities

- Replace the proposal’s generic “existing approaches” section with the cited summary in `docs/related_work.md`.
- Train models with and without `G1`/`G2`.
- Quantify whether engineered interaction features improve accuracy and explanation quality.
- Use the scenario simulator to measure average predicted lift from top-k feasible interventions.
- Use the planner to produce a concise support plan for each at-risk student.
