# Revised Project Outline

## 1. Revised Problem Statement

The original proposal framed the task as predicting final student grade `G3` on the UCI Student Performance dataset. The stronger version of the project is:

**Build an interpretable early-warning system that predicts student outcomes, simulates feasible changes to actionable factors, and generates a personalized intervention plan that educators can use.**

That framing directly addresses the TA’s concern that prediction alone does not create value.

## 2. How The TA Feedback Changes The Project

| TA feedback | Design response in this scaffold |
| --- | --- |
| “Standard tools on a standard dataset” | Keep the dataset, but shift novelty toward actionability, recourse, and educator support. |
| “Does not translate predictions into actions” | Add a scenario simulator that estimates how changing specific features affects predicted outcomes. |
| “Explore feature engineering techniques” | Add interaction features and compare runs with and without them. |
| “Add an agentic layer” | Add an intervention planner that converts predictions plus simulated scenarios into a personalized support plan. |
| “No work has been cited” | Add a related-work summary and bibliography grounded in student performance prediction, interpretability, and actionable recourse. |

## 3. Revised System Architecture

### 3.1 Prediction Layer

- Predict `G3` as a regression target.
- Optionally derive a pass/fail target using `G3 >= 10`.
- Evaluate models both **with** and **without** `G1` and `G2`.
- Start with interpretable baselines (`Ridge`, `Lasso`, decision tree), then compare against stronger tree ensembles if time allows.

### 3.2 Feature Engineering Layer

Interaction features are now a first-class deliverable rather than an optional extra. Initial candidates:

- `studytime x absences`
- `studytime x failures`
- `schoolsup x failures`
- `internet x studytime`
- `Medu x Fedu`

These are useful because they encode relationships that raw columns alone can miss, while still staying explainable.

### 3.3 Scenario Simulation Layer

For each at-risk student, the project should simulate feasible changes to actionable features such as:

- increasing study time
- reducing absences
- enabling school support
- enabling family support
- improving access-related factors such as internet support

The output should be a ranked list of scenarios like:

- “If absences drop by 3 and school support changes from no to yes, predicted grade increases from 8.4 to 10.1.”

This is the part that turns the model from a predictor into a planning tool.

### 3.4 Intervention Planning Layer

The intervention layer should take:

- predicted grade and risk band
- top contributing risk factors
- best simulated improvement scenarios

and turn them into:

- a short educator-facing summary
- 2-4 recommended interventions
- a monitoring plan for the next check-in

The scaffold includes a rule-based planner now, plus a prompt-building path for a later LLM-backed agent.

## 4. Research Questions

1. How accurately can we predict `G3` using demographic, social, behavioral, and academic features?
2. Do interaction features improve predictive performance and interpretability over a plain baseline?
3. How much predicted uplift can be achieved through feasible changes to actionable features?
4. Can a structured intervention planner produce more useful educator-facing outputs than raw model scores alone?

## 5. Experimental Plan

### Experiment A: Baseline Prediction

- Compare baseline regressors on the Portuguese dataset.
- Metrics: RMSE, MAE, `R^2`.

### Experiment B: Usefulness Of Prior Grades

- Train one set of models with `G1`/`G2`.
- Train another without them.
- Report the tradeoff between accuracy and early-warning usefulness.

### Experiment C: Feature Engineering

- Compare baseline features versus baseline + interaction features.
- Analyze whether interactions improve both metrics and explanation quality.

### Experiment D: Scenario Simulation

- For students below the pass threshold, generate top-k feasible scenarios.
- Report:
  - percentage of students with at least one feasible improvement path
  - average predicted grade lift from the best scenario
  - most common recommended feature changes

### Experiment E: Intervention Planning

- Generate intervention plans for a sample of students.
- Evaluate quality using:
  - actionability
  - specificity
  - alignment with top risk factors
  - educator readability

## 6. What Counts As The Project Contribution

The project’s contribution is no longer “we trained a few models on UCI student data.” It becomes:

- an interpretable student-risk model
- an actionable recourse/simulation module
- a personalized intervention planner
- an evaluation of whether feature engineering improves both accuracy and usefulness

That is substantially better aligned with the course feedback.

## 7. Suggested Report Structure

1. Introduction and motivation
2. Related work on student performance prediction, interpretability, and recourse
3. Dataset and preprocessing
4. Feature engineering strategy
5. Predictive modeling
6. Scenario simulation for actionable changes
7. Intervention planner design
8. Results and discussion
9. Limitations, ethics, and future work

## 8. Milestones

### Milestone 1

- finalize dataset loading and preprocessing
- implement baseline models
- reproduce results with and without `G1`/`G2`

### Milestone 2

- add interaction features
- run ablation study
- add feature importance / explanation outputs

### Milestone 3

- implement scenario simulator
- define actionable feature constraints
- evaluate top-k uplift scenarios

### Milestone 4

- implement intervention planner
- generate sample plans
- polish results and narrative for the final report

## 9. Seed References

See `docs/related_work.md` and `refs/seed_papers.bib`.
