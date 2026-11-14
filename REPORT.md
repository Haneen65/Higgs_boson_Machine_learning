# Higgs Boson Classification — Report

## 1. Problem
Classify ATLAS collision events as signal (Higgs → ττ) or background, judged by the Approximate Median Significance
AMS = √{2[(s + b + 10)·ln(1 + s/(b + 10)) − s]}, where *s* and *b* are the sums of **event weights** of correctly and wrongly selected events.

**Data.** Official `training.csv` (250,000 events, 30 features, true `Weight`, label) and `test.csv` (550,000 events). The class weight totals are ≈ 691.99 (signal) and ≈ 410,999.85 (background). (An earlier run on a weight-less copy with constant per-class proxy weights was replaced by this one; several weight-related conclusions changed, see §5–6.)

## 2. Data and EDA — what mattered
* No duplicates, no true NaNs; `-999.0` marks undefined values. 34.3 % of events are signal by count, but signal carries only ≈ 0.17 % of the weight: counts and weights tell very different stories, and one wrongly selected background event can cost far more than a correct signal event gains.
* **`-999` is structural.** Seven jet-pair features are undefined for exactly the events with fewer than two jets (71 %), three leading-jet features when there is no jet (40 %); `DER_mass_MMC` fails for 15 % of events (21.5 % of background but only 3.3 % of signal). Events with undefined mass are only 7.4 % signal vs 39 % otherwise — the missingness itself is informative.
* Signal share depends on jet number (25 % / 36 % / 51 % / 30 % for 0–3 jets).
* No single feature separates well (best one-feature AUC 0.74); angles are at AUC ≈ 0.50. `DER_mass_MMC` has a narrow signal peak near 125 GeV (std 28 vs 70 GeV for background) — a shape a monotone AUC underrates.
* Redundancy is physical (e.g. `PRI_jet_all_pt` ~ `DER_sum_pt`; `DER_mass_MMC` ~ `DER_mass_vis`, ρ = 0.86); no feature was dropped automatically.

## 3. Preparation and validation
* `EventId`, `Weight`, `Label` are never features. `-999` → NaN for boosting; median imputation (+ optional indicators), scaling and one-hot of `PRI_jet_num` live **inside a Pipeline/ColumnTransformer**.
* **Validation:** one stratified 80/20 split (seed 42) for every model; key findings re-checked on seeds 7 and 2024. Validation weights are rescaled by N/n to full-training totals (as in the competition starting kit). The test file was never used for decisions.
* **Metric reporting:** picking the best cut on the same data that reports AMS is optimistic and AMS is spiky, so I report *AMS_oracle*, *AMS_honest* (cut chosen on one half of validation, scored on the other) and **AMS_smooth** (maximum of a bootstrap-averaged AMS-vs-cut curve). ROC-AUC is reported alongside and served as tie-breaker (stable to ± 0.002 across splits; AMS_smooth varies by ≈ 0.1).
* My AMS implementation is unit-tested against the competition formula.

## 4. Models and results (seed-42 split)
| Model | AUC | AMS_smooth | AMS_honest |
|---|---|---|---|
| Dummy | 0.500 | 0.68 | 1.08 |
| Logistic regression | 0.815 | 2.06 | 2.04 |
| Decision tree (depth 8) | 0.882 | 2.91 | 2.92 |
| Extra trees | 0.889 | 3.15 | 3.09 |
| Random forest | 0.905 | 3.42 | 3.39 |
| HistGradientBoosting | 0.909 | 3.47 | 3.48 |
| LightGBM | 0.910 | 3.51 | 3.36 |

The big step is linear → non-linear (+0.09 AUC, AMS 2.06 → 3.4+). Among RF / HGB / LightGBM the AMS gaps (≈ 0.1) are inside the split noise. Over three splits (AUC / AMS_smooth, mean ± sd): LightGBM+eng 0.912 / 3.59 ± 0.04, HGB 0.912 / 3.59 ± 0.10, HGB+eng 0.912 / 3.57 ± 0.08, RF 0.907 / 3.55 ± 0.11, HGB balanced weights 0.907 / 3.44 ± 0.05, LogReg 0.817 / 2.06 ± 0.01.

## 5. Experiments (full table: `outputs/experiments.csv`)
| # | Question | Result | Kept? |
|---|---|---|---|
| 7.1 | How to handle `-999`? | Keep value / NaN / NaN+indicator ≈ identical (AUC 0.909). Dropping the 7 sparse columns: AUC −0.0036, AMS 3.47 → 3.43 (noise); LogReg −0.012 AUC. Median imputation creates a spike for 15 % of events. | NaN-native, no dropping |
| 7.2 | Sample weights? | **Raw weights:** HGB AUC 0.909 → 0.895 but AMS 3.47 → 3.52 (not lower; one split). **Balanced weights:** AUC 0.903, AMS 3.37; RF AMS 3.42 → 2.65; 3-split HGB AMS 3.44 vs 3.59. | No training weights |
| 7.3 | Physics features? | HGB AUC 0.9093 → 0.9098, AMS 3.47 → 3.50; LogReg 0.8152 → 0.8186. Over 3 splits AUC +0.0006, AMS 3.57 vs 3.59 | Not used |
| 7.4 | All vs subset? | Dropping 5 azimuth angles: no loss; top-20 / top-12: AUC 0.9091 / 0.9071 | All 30 kept |
| 7.5 | Tuning? | Inner-split search: +0.0008 AUC, same AMS | Tuned config |
| 7.6 | Ensemble? | HGB+LightGBM +0.0004 AUC, AMS 3.53 (single 3.49 / 3.55); adding LogReg: AUC 0.904 | No |
| 7.7 | AUC vs AMS? | Correlated (ρ ≈ 0.83) but not equal: raw-weight HGB has lower AUC than balanced-weight HGB yet higher AMS | Report both |

**What did not work as expected.** (1) Class-balanced weights, which I expected to help for a rare signal, lowered AUC and AMS for HGB, RF and LogReg. (2) Raw weights lowered AUC clearly but not AMS: AUC and AMS ranked these models differently. (3) The physics features raised AUC slightly on one split, but gave no AMS gain over three splits.

**Weights: proxy vs real.** An earlier run with constant proxy weights suggested "weights never help" and put the best cut at about 1 % of events. With the real weights the effect of raw weights on AMS is neutral-to-slightly-positive and the best cut is about 14 %. Weight conclusions from proxy weights should not be reused.

## 6. Threshold selection
At `p ≥ 0.5` (31 % of events labelled signal) accuracy is 0.842 but AMS only 2.87. The best smoothed cut on the validation split is the top ≈ 14 % of events (AMS 3.47, accuracy 0.780, only 1.5 % of background selected, 39 % of signal recalled); the AMS curve is broad around it (≈ 3.4–3.5 for 12–20 %).

The final cut was chosen on 5-fold **out-of-fold** predictions of all 250,000 training events: the *middle of the plateau* of cuts whose bootstrap-smoothed AMS is within one standard deviation of the best (12.2 %–18.4 %), i.e. **q = 15.3 %** (the argmax was 13.4 %). The fraction, not a raw score, is used so the rule survives refitting. Caveat: for this step weights were scaled to test size by 550,000/250,000 (not verified; the starting kit scales validation to the training totals), so the OOF AMS (≈ 5.3) is not a number to compare with the leaderboard; the validation-scale AMS (≈ 3.5) is. The validation optimum (14 %) lies inside the plateau, so the threshold is not very sensitive to this choice.

## 7. Interpretation and error analysis
* Permutation importance (clue, not proof): `DER_mass_MMC` is by far the most important (AUC drop 0.10, four times the next: `DER_mass_vis` 0.025, `DER_mass_transverse_met_lep` 0.021) — consistent with the Higgs mass peak.
* At `p ≥ 0.5`: recall 0.73, 20.6 % of selected events are background; false-positive weight ≈ 27,600 vs true-positive weight ≈ 477. False positives lie under the same mass peak as true positives; false negatives look more background-like. Recall is lowest for 0-jet events (0.62) and highest for 2-jet events (0.85); when `DER_mass_MMC` is undefined recall is 0.34 (vs 0.74). **0-jet events carry most of the false-positive weight (≈ 15,300 of 27,600)** — the category where the model is weakest and mistakes cost most. Hypothesis for next time: separate treatment per jet category.

## 8. Recommendation, limitations, next steps
**Recommendation:** HistGradientBoosting on the 30 original features (`-999 → NaN`, no sample weights; 400 iterations, learning rate 0.05, 63 leaves, `min_samples_leaf` 40), out-of-fold AUC 0.9115, validation AMS ≈ 3.5; label the top 15.3 % of events as signal.

**Limitations.** (1) Test-size weight factor used for the threshold is an unverified assumption (low sensitivity); (2) AMS is noisy: differences below ≈ 0.1 between strong models are not evidence; (3) the submission was not scored on Kaggle and the `RankOrder` direction ("1 = most background-like") comes from memory — check it against the official sample file; (4) no neural network/SVM, no systematic uncertainties, no repeated CV; (5) explanations of *why* (weights, error patterns) are hypotheses from one split.

**Next steps.** Repeated stratified CV including the raw-weight experiment; per-jet-category models; features for undefined mass; verify test weight normalisation and `RankOrder`, then re-select the threshold.

*References:* [Kaggle competition](https://www.kaggle.com/competitions/higgs-boson/overview) and data; [CERN Open Data record 328](https://opendata.cern.ch/record/328); Adler et al., "The Higgs boson machine learning challenge", PMLR 42 (2015). The notebook is original work.
