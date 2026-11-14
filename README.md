# Higgs Boson Machine Learning Challenge

Classifying ATLAS collision events as Higgs **signal** (`s`) or **background** (`b`) and evaluating with the competition metric, Approximate Median Significance (AMS), using event weights.

> **Data.** The reported results use the **official** `training.csv` (250,000 events, true `Weight`) and `test.csv` (550,000 events), which are not included in the repository (see [`data/README.md`](data/README.md)). If those files are missing, `src/higgs/data.py` falls back to a weight-less public copy with constant per-class proxy weights; results from that mode are only good for model-vs-model comparison, and weight-related conclusions differ from the official run.

## Contents

| Path | What |
|---|---|
| `notebooks/higgs_boson_analysis.ipynb` | Full story: data understanding → EDA → preprocessing → validation → baselines → 8 experiments → error analysis → final model → submission. Executed; runs top to bottom in ≈ 20 min on 2 CPU cores. |
| `REPORT.md` | Concise written report (findings, decisions, limitations, recommendation). |
| `presentation/` | Slides (see below). |
| `src/higgs/` | `data.py` (loading, official vs fallback weights), `metrics.py` (AMS, threshold search, honest and bootstrap-smoothed AMS), `features.py` (`-999` strategies, engineered features). |
| `tests/` | Unit tests: our AMS equals the competition formula, curves match direct computation. |
| `outputs/` | `experiments.csv` (experiment table), `seed_stability.csv`, `permutation_importance.csv`, `figures/` (all plots). |
| `submission/submission.csv` | `EventId, RankOrder, Class` for the test file. |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q tests                                   # AMS implementation checks
cd notebooks && jupyter nbconvert --to notebook --execute --inplace higgs_boson_analysis.ipynb
```
Everything is seeded (`SEED = 42`; extra splits 7 and 2024). Developed with Python 3.13, scikit-learn 1.9, pandas 3.0, LightGBM 4.7.

## Headline results (validation, official weights)

| Model | ROC-AUC | AMS (smoothed) |
|---|---|---|
| Dummy (no learning) | 0.500 | 0.68 |
| Logistic regression | 0.815 | 2.06 |
| Decision tree (depth 8) | 0.882 | 2.91 |
| Extra trees | 0.889 | 3.15 |
| Random forest | 0.905 | 3.42 |
| HistGradientBoosting | 0.909 | 3.47 |
| LightGBM | 0.910 | 3.51 |

Single split, seed 42. AUC is stable across splits (± 0.002); AMS_smooth varies by about 0.1, so differences inside the strong-model group are not evidence (see `outputs/seed_stability.csv`).

**Final model:** HistGradientBoosting, 30 original features, `-999 → NaN`, no sample weights; out-of-fold AUC 0.9115; decision threshold = top **15.3 %** of events by score (middle of the plateau of near-optimal cuts, chosen on out-of-fold training predictions only).

## Main lessons
1. Non-linear trees beat a linear model by a wide margin (AUC 0.91 vs 0.82; AMS 3.5 vs 2.1); differences among strong tree models are small.
2. `-999` is structural (jet multiplicity) and informative; median imputation invents a spike, and NaN-native handling is enough.
3. Accuracy is the wrong target: the AMS-best cut (top ~14 %) has lower accuracy (0.78) than `p ≥ 0.5` (0.84) but a higher AMS (3.47 vs 2.87).
4. Class-balanced weights hurt; raw weights lowered AUC but not AMS (AUC and AMS ranked these models differently); engineered features gave no reliable AMS gain.
5. 0-jet events carry most of the false-positive weight; per-jet-category modelling is the natural next step.

## Sources and attribution
* Competition: <https://www.kaggle.com/competitions/higgs-boson/overview>; data: <https://opendata.cern.ch/record/328>.
* Adler et al., *The Higgs boson machine learning challenge*, PMLR 42 (2015) — metric definition. The AMS formula was checked against the starting-kit expression.
* The fallback data file (not needed with the official files) is a public copy of the competition training events from the GitHub repository `patrontheo/ml-project-1-pml` (fallback data only; none of its code was used). All modelling code in this repository is original.
