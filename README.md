# Training-Optimizer

A single-file in python that combines running, strength training, and user defined daily metrics to optimize training plans and predict the chances of successfully achieving goals.

The engine implements over statistical methods drawn from sports science literature (Banister 1975, Foster 1998, Gabbett 2016, Jack Daniels VDOT, etc.) and presents them in a 14-tab Tkinter dashboard.

## Features
The codebase is organized into 18 functional sections, all wired together by a single `FitnessOptimizer` orchestrator class.

### Modeling & prediction
- **Banister two-component impulse-response model** simulates fitness and fatigue dynamics from your training history, with Levenberg-Marquardt parameter fitting to personalize the time constants (`BanisterModel`).
- **Bayesian-style linear regression** projects your improvement trajectory with confidence intervals and estimates the probability of hitting your goal by the target date (`PredictionEngine`).
- **Adaptation quality multiplier** applies a sigmoid discount to incoming training stimulus based on accumulated fatigue, so sessions logged on top of heavy fatigue contribute less than the same sessions in a recovered state.
- **Overreaching classification** labels each training state as normal, functional, or non-functional overreaching based on the current adaptation quality.
- **Taper optimization** runs a grid search across taper patterns (linear, exponential, and step) to find the schedule that maximizes predicted race-day performance.
- **What-if simulator** generates performance prediction curves with and without a proposed session, so you can see how a workout would affect fitness, fatigue, and peak timing before actually doing it.

### Load & assessment quantification
- **TRIMP** quantifies running load using HR-weighted exponential scaling, with an RPE proxy fallback when heart-rate data isn't available (Banister).
- **Tonnage** quantifies lifting load as the total weight moved across all sets and reps, with optional session-RPE weighting.
- **VDOT** normalizes any race distance and time into a single fitness score for cross-distance comparison (Jack Daniels).
- **Epley 1RM** estimates your one-rep max from any rep-max performance.
- **Z-scores** express each assessment as standard deviations above or below your personal baseline, so a 5K time and a squat 1RM land on the same scale.

### Analytics
- **Lagged cross-correlation** measures the Pearson correlation between daily metrics and performance at lags from 0 to 14 days, surfacing which inputs matter and how long their effect takes to show up.
- **Granger causality** runs a simplified F-test against an autoregressive baseline to check whether a metric actually helps predict performance beyond performance's own history.
- **Benjamini-Hochberg FDR correction** controls the false-discovery rate when testing many metrics at once, so chance correlations don't get reported as real findings.
- **ACWR** (Acute:Chronic Workload Ratio) uses EWMA to compare recent training load against your chronic baseline, classifying the result into safe, caution, and high-risk zones.
- **Foster monotony and strain** quantify how repetitive your training has been, since uniform high-load weeks raise injury and illness risk independent of total volume.
- **Logistic injury risk model** estimates the probability of injury in the next seven days from ACWR, monotony, strain, fatigue, soreness, and readiness trend, intentionally not personalized (reasoning in Limitations & Disclaimers).
- **Concurrent training interference detection** runs Welch's t-test or Mann-Whitney U with Cohen's d to test whether heavy lifting before running (or vice versa) measurably impairs your performance.
- **Partial correlation** isolates whether one training domain hurts another after controlling for shared volume, useful for detecting goal conflicts that wouldn't show up in raw correlation.

### Personalization & adaptation
- **Per-feature blending tracker** lets each feature personalize at its own data threshold (between 15 and 30 sessions), so fast-learning components don't have to wait for slow-learning ones.
- **Ridge regression** learns personal readiness weights from your own pre-session metrics and outcomes once you have 30 or more logged sessions.
- **CUSUM drift detection** uses a warmup period and then watches for systematic prediction errors, triggering a Banister refit when the model's view of you starts diverging from reality.
- **EWMA bias tracking** monitors prediction residuals and applies small online nudges to the fatigue parameter (k₂) between full refits, keeping the model honest without expensive retraining.
- **Plateau detector** runs a slope-significance test on your recent assessments to flag when improvement has stalled or started declining, so you know when it's time to change the training stimulus.
- **Thompson Sampling bandit** learns which session types tend to produce improvement for you by maintaining a Beta posterior over each option, sampling from those posteriors to recommend the next session, layered on top of the rule-based plan as a Phase-2 personalization step.

### Diagnostics
- **KNN with Gower distance** scores prediction confidence by finding similar past sessions across mixed feature types, since predictions about novel session contexts deserve less trust than ones with lots of similar precedent.
- **CUSUM-based changepoint detection** finds natural mesocycle boundaries in your training-load history by flagging sustained shifts in daily load.
- **Cohen's d effect size with paired or independent t-tests** compares two training blocks to quantify both whether the difference between them is statistically significant and how meaningful it actually is.
- **Shapley-style race post-mortem** attributes the gap between predicted and actual assessment performance to individual metrics, weighting each contribution by its optimal lag.
- **Modified Z-score outlier detection** flags unusual metric entries using median and median absolute deviation (MAD) instead of mean and standard deviation, so a single bad value doesn't distort the threshold for the rest.
- **Composite data-quality score** combines logging rate, metric rate, completeness, and outlier rate into a single grade between 0 and 1, with suggestions for which habit to improve next.


### GUI
 
A 14-tab Tkinter dashboard built on top of the engine:
 
1. **Dashboard** shows readiness, injury risk, ACWR, personalization, and weekly volume.
2. **Goals** lays out milestones with logarithmic spacing alongside conflict analysis.
3. **Running Log** displays running session tables.
4. **Lifting Log** displays lifting session tables.
5. **Assessments** shows VDOT and 1RM history with normalization.
6. **Predictions** plots your improvement trajectory with 68% and 95% confidence bands.
7. **Correlations** shows per-metric lag plots with FDR-survival labels.
8. **Simulator** lets you propose interactive what-if sessions.
9. **Readiness** breaks down each component live with adjustable inputs.
10. **Injury Risk** decomposes risk factors and shows ACWR history.
11. **Adaptive Plan** combines rule-based suggestions with Thompson Sampling stats.
12. **Block Analysis** covers block comparison, changepoints, and post-mortem.
13. **Data Quality** shows per-feature personalization bars, a drift demo, and an interference test.
14. **Log Data** provides forms for adding sessions, assessments, and metrics; submitting refreshes every other tab.

## Architecture
 
Single file, organized top-down:
 
```
training_optimizer_v2.py
├── Enums & constants                          (GoalType, RunWorkoutType, defaults, ReadinessBand, ...)
├── Section 1   Goal setting & milestones      (Goal, GoalEngine, logarithmic spacing)
├── Section 2   Running session log            (RunningSession, TRIMP)
├── Section 3   Lifting session log            (LiftingSession, LiftingExercise, tonnage)
├── Section 4   Custom metrics schema          (CustomMetricDefinition, MetricEntry)
├── Section 5   Assessment normalization       (Assessment, AssessmentNormalizer, VDOT, Epley)
├── Section 6   Prediction & projection        (PredictionEngine, Bayesian-blended regression)
├── Section 7   Correlation & insight          (CorrelationEngine, lagged Pearson, Granger, BH-FDR)
├── Section 8   Adaptive training plan         (AdaptivePlan, ThompsonBandit)
├── Section 9   Banister model + simulator     (BanisterModel, ACWR, taper, what-if)
├── Section 9g  Prediction confidence          (SimilarityScorer, KNN + Gower)
├── Section 10  Personalization progression    (PersonalizationTracker, per-feature blending)
├── Section 10+ Model-health monitors          (DriftDetector, PlateauDetector)
├── Section 11  Daily readiness score          (ReadinessScore, ridge personalization)
├── Section 12  Injury risk prediction         (InjuryRiskModel, logistic, population-based)
├── Section 13  Interference detection         (InterferenceDetector, Welch / Mann-Whitney, Cohen's d)
├── Section 14  Goal conflict detection        (GoalConflictDetector, matrix + partial correlation)
├── Section 15  Data quality scoring           (DataQualityScorer, modified Z + composite score)
├── Section 16  Retrospective block analysis   (BlockAnalysis, Cohen's d + CUSUM changepoints)
├── Section 17  Post-session feedback loop     (FeedbackLoop, EWMA bias + parameter nudges)
├── Section 18  Race/assessment post-mortem    (PostMortem, linearized Shapley)
├── FitnessOptimizer                           (orchestrator)
└── FitnessOptimizerGUI                        (Tkinter dashboard, 14 tabs)
```
 
Section 8 (`AdaptivePlan`) and the Section 10 supplement classes (`DriftDetector`, `PlateauDetector`) appear later in the source file than their numerical order suggests, due to their dependencies on classes defined earlier. The architecture tree above shows them in numerical order for readability.

## Limitations & Disclaimers
- **Not medical advice** This is an analysis tool, not a clinical or coaching service. Predictions are statistical estimates with substantial uncertainty, especially with limited data.
- **The injury risk model is population-based and intentionally not personalized** Users normally do not have enough injury events to train a logistic model without overfitting. Coefficients come from published literature (Gabbett 2016, Foster 1998).
- **Granger causality and BOCPD are simplified versions** For research grade work, swap in `statsmodels.tsa.stattools.grangercausalitytests` and a true Bayesian Online Changepoint Detection implementation (Adams & MacKay 2007).
- **Bayesian linear regression** Here is OLS with precision-weighted blending against a prior, not a full posterior. A real Bayesian implementation would use PyMC or Stan.
- **Synthetic demo data** The bundled 6-month dataset is generated from a fixed-seed RNG; correlation/post-mortem outputs in the GUI reflect that synthetic structure, not real-world findings.
- **Personalization takes time** Most features need 30 sessions before fitted parameters carry meaningful weight against the sport-science defaults.
