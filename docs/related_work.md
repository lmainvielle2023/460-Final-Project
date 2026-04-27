# Related Work Notes

This summary is intended to replace the uncited “existing approaches” section from the proposal with a short, defensible literature review.

## Student Performance Prediction

**Cortez and Silva (2008)** is the canonical source behind the UCI dataset used in this project. It shows that student performance can be modeled with demographic, social, and school-related variables, and it also highlights an important caveat: `G1` and `G2` are strongly predictive but reduce the usefulness of truly early intervention.

**Hellas et al. (2018)** surveys the broader literature on academic performance prediction and argues that the field often lacks strong validation and comparable reporting. That directly supports a cleaner experimental design in this project.

**Namoun and Alshanqiti (2021)** review student performance prediction through data mining and learning analytics. Their survey emphasizes that regression, neural, and tree-based methods are common, while explanatory analytics and validation across settings remain limited. That gap supports the decision to focus on interpretability and actionability instead of adding another black-box model.

## Interpretability And Early Warning

**Lundberg and Lee (2017)** provides the SHAP framework, which is a standard way to explain model predictions at both the global and local level. This is useful for identifying the major factors behind a specific student risk score.

**Xia and Qi (2023)** moves from early warning toward interpretable early warning recommendations. The key lesson for this project is that educational prediction becomes more valuable when it is tied to understandable next steps rather than raw scores alone.

## Actionable Recommendations And Recourse

**Wachter, Mittelstadt, and Russell (2018)** argues that counterfactual explanations can tell affected people what would need to change to reach a better outcome. That idea maps well onto the TA’s suggestion to simulate how changing specific features can affect a student outcome.

**Ustun, Spangher, and Liu (2019)** formalizes actionable recourse, meaning recommendations should focus on variables a person or institution can realistically change. For this project, that means separating immutable attributes from intervention-relevant features like support, attendance, or study time.

## How This Project Builds On The Literature

The revised project combines four ideas from the papers above:

1. Use the well-known student performance benchmark responsibly rather than claiming dataset novelty.
2. Keep models interpretable enough to explain individual risk.
3. Add counterfactual or recourse-style simulations over actionable features.
4. Present educator-facing intervention plans instead of only reporting predicted grades.

## Suggested Citations To Use In The Report

- Cortez, P., and Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*.
- Hellas, A. et al. (2018). *Predicting Academic Performance: A Systematic Literature Review*.
- Namoun, A., and Alshanqiti, A. (2021). *Predicting Student Performance Using Data Mining and Learning Analytics Techniques: A Systematic Literature Review*.
- Lundberg, S. M., and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*.
- Wachter, S., Mittelstadt, B., and Russell, C. (2018). *Counterfactual Explanations Without Opening the Black Box*.
- Ustun, B., Spangher, A., and Liu, Y. (2019). *Actionable Recourse in Linear Classification*.
- Xia, X., and Qi, W. (2023). *Interpretable Early Warning Recommendations in Interactive Learning Environments*.
