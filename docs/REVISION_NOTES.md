# Revision notes: from the course submission to v2

The original project was submitted for the University of Twente *Explainable AI* course
(June 2025; report and Co-12 tables in [`original_submission/`](original_submission/)).
This revision addresses the grader's feedback point by point and fixes problems found
while re-running the original code.

## Feedback → changes

| Feedback | What changed |
|---|---|
| **Problem context lacks strong motivation; importance justified with one reason** | The introduction now argues from three angles: (1) the *user* — an author choosing a venue needs to know whether a suggestion rests on the paper's substance or on surface words, and venue mismatch is a common cause of desk rejection; existing venue recommenders (Yang & Davison 2012; Pradhan & Pal 2019; Kobs et al. 2020) show that the problem is real; (2) the *developer* — explanations are used to find shortcuts and data errors, which we demonstrate (*Front Matter* → VLDB); (3) the *method* — post-hoc explainers are mostly evaluated anecdotally (Nauta et al. 2023) and disagree with each other (Krishna et al. 2022); our setting allows an exact ground truth, making it a controlled test of those explainers. |
| **Research questions present but not very novel** | Reframed around questions that need this setting to answer: RQ1 faithfulness against exact Shapley values, RQ2 stability / continuity / disagreement, RQ3 shortcut discovery, RQ4 whether the model's confidence can be trusted and acted on. |
| **Methods suitable with some rationale** | Every method now has an explicit role and rationale (README table, `REPORT.md` §4), including why explanations are computed over the lemmas the model actually sees and why removal is the perturbation. |
| **Pipeline reasonably structured / partially reproducible** | One installable package (`src/tcner_xai`) used by six notebooks; fixed seeds everywhere; pinned `requirements.txt`; data in the repo; `scripts/run_notebooks.py` re-executes everything; cached artefacts are rebuilt automatically. The Anchors cell that crashed in the original (missing `alibi`) now runs. |
| **2+ techniques compared with basic analysis** | A quantitative benchmark on 1,000 test titles compares LIME, KernelSHAP, occlusion, feature-space SHAP and coefficient × input against exact Shapley values at equal model-call budgets: rank correlation, top-k agreement, sign agreement, comprehensiveness / sufficiency, seed stability, continuity, contrastivity, runtime, plus a pairwise disagreement matrix. Global methods are compared with rank correlation and top-20 overlap. |
| **Evaluation criteria (Co-12) applied to most techniques with some reflection** | All **12** Co-12 properties (the original covered 9) for **11** methods (originally 7). Every rating that can be measured (40 of 132 cells) is derived from an experiment in this repo with fixed thresholds in code; the evidence for every cell, measured or reasoned, is saved to `results/co12_evidence.csv`. |
| **Slight modifications to the baseline** | New contributions: exact word-level Shapley values as ground truth; equal-budget convergence study; a null-word continuity test; anchor precision/coverage re-measured on real titles; minimal-edit counterfactuals; calibration with temperature scaling, prior-shift correction and selective prediction; data audit that traces global features back to data artefacts. |
| **Integrates 1–2 additional papers / tools** | 20+ references now, each tied to a concrete use (see `REPORT.md` references); new tools: KernelSHAP, alibi Anchors, ERASER metrics, ECE / temperature scaling, EM prior adjustment. |
| **Limitations identified with brief discussion** | A full limitations / threats-to-validity section: the removal game as *one* definition of ground truth, the linear model as a best case for explainers, a single dataset and test distribution, rating thresholds as a design choice, no user study, and data issues. |

## Errors in the original analysis that were corrected

1. **LIME and Anchors explained a different model.** They passed raw titles to the
   vectoriser, skipping the spaCy lemmatisation used in training. This changes the predicted
   venue for 283 of 3,373 test titles (8.4%) and lowers accuracy from 86.2% to 82.8%.
2. **Anchors never ran.** The cell failed with `ModuleNotFoundError: alibi`, so the Anchor
   rules and numbers in the report were not backed by the notebook.
3. **SHAP force plots were misread.** Feature-space SHAP attributes about half of each
   explanation to n-grams that are *absent* from the title. The report's reading that
   “*web* was associated with WWW” for a title without the word *web* described the effect of
   the word's absence.
4. **“Overconfidence” was the wrong conclusion.** Aggregate calibration shows the model
   is *under*-confident (mean confidence 0.71 vs accuracy 0.86, ECE 0.15); a handful of
   confident errors is anecdotal evidence. Temperature scaling fitted on training folds
   reduces ECE to 0.02.
5. **The surrogate tree was not a reasonable approximation.** Its fidelity was never
   measured; the depth-4 tree agrees with the model on ≈ 70% of test titles and recovers none
   of its SIGGRAPH predictions. The report also described a different depth and sample size
   than the code used.
6. **Anchor coverage was misinterpreted.** alibi's coverage is measured on its own
   perturbation samples; on real test titles the rule `network → INFOCOM` holds for only
   about half of the titles containing *network*.
7. Smaller issues: no random seeds (LIME samples changed every run), a background set made
   of the first 100 training rows, deprecated `multi_class` argument, `Fig. ??` broken
   references in the report, and boilerplate / HTML artefacts in the data that went
   unnoticed.
