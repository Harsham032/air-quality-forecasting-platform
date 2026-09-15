# Methodology

This document states what the pipeline does and, where a common alternative
exists, why it was not taken. Every choice below either prevents a specific
error or makes one visible.

## 1. The dataset's missing values are not values

The UCI Air Quality file encodes a missing reading as `-200`. Nothing in the
file marks it as special: it is a number in a numeric column, and every library
that reads the file will happily average it, scale it, and train on it.

`-200` appears in **16,701 cells across 8,530 of the 9,357 rows** — 91.2% of the
series contains at least one. The consequences of reading it as a measurement
are not subtle:

| Column | Mean if `-200` is read as data | Mean over observed hours |
| --- | ---: | ---: |
| `CO(GT)` | −34.21 | 2.15 |
| `NO2(GT)` | 58.15 | 113.09 |
| `NOx(GT)` | 168.62 | 246.90 |
| `AH` | −6.84 | 1.03 |

A negative mean carbon monoxide concentration is not a small inaccuracy. It is
physically impossible, and it is what the arithmetic produces.

`aqf.data.loader` replaces the sentinel with `NaN` before anything else touches
the frame, and `missingness_report` prints both columns of the table above so
the difference is visible rather than assumed.

### Why this is worse than ordinary missing data

Min-max scaling is the step that converts the problem from *wrong* to
*invisible*. `MinMaxScaler` maps the observed range onto `[0, 1]`. With the
sentinel present, the range of `NO2(GT)` is `[-200, 333]` instead of
`[2, 333]`. The real measurements are compressed into the top 62% of the scale,
and the genuine hour-to-hour variation — the entire signal — is squeezed towards
a narrow band while the missing hours sit at exactly 0.

A model trained on that is being asked to predict a spike train. It cannot, and
its error metrics are computed on the same distorted axis, so nothing in the
output announces the problem.

## 2. Columns that barely exist are dropped, not imputed

`NMHC(GT)` is observed in **9.8%** of hours, and its longest unbroken gap is
**8,126 hours — 339 days**. Imputing that is not filling in missing data; it is
generating a year of synthetic measurements and calling them a feature.

Any column observed less than `min_observed_share` of the time (default 0.5) is
dropped and *reported as dropped*. On the real file this removes `NMHC(GT)` and
nothing else.

## 3. Interpolation is bounded, and counted

Short outages in an hourly pollutant series are reasonable to bridge: the
underlying process is continuous and a two-hour gap is well constrained by its
neighbours. A two-day outage is not.

The loader interpolates on the time index with `limit=6` hours and
`limit_area="inside"`, then **drops every row that still has a gap**. On the
real file this bridges 1,693 cells and drops 1,737 rows, leaving 7,620 hours
from 2004-03-10 18:00 to 2005-04-04 14:00.

Both numbers are reported. A "clean" series is less trustworthy than a smaller
one when you cannot tell how much of it was reconstructed.

## 4. The split is chronological, and the scaler sees only the past

Two separate leaks are possible here, and both are easy to introduce with
`train_test_split`.

**Shuffling.** A random split puts hour *t+1* in training and hour *t* in test.
The model is then evaluated on interpolating between values it has already seen,
which is a much easier problem than forecasting and scores far better.
`split_by_time` divides the series by position: the first 70% trains, the next
15% validates, the final 15% tests.

**Scaler leakage.** Fitting a scaler before splitting lets the training data
inherit the test period's mean and range. `split_by_time` computes the mean and
standard deviation on the training slice only. `tests/test_windows.py` asserts
this directly: doubling every value after the training cut must not move the
training statistics by so much as a float.

**Boundary windows.** Windows are built *within* each part, not across the
series and split afterwards. Otherwise the first `lookback` windows of the test
set contain training hours. This costs `lookback + horizon - 1` windows at each
of the two boundaries, and `tests/test_windows.py` asserts exactly that count.

## 5. Errors are reported in micrograms per cubic metre

Predictions are inverted back to the target's own units before they are scored.
`evaluate_forecast` refuses to score a target that still looks scaled — if the
truth lies entirely within a unit interval, it raises rather than returning a
number.

This is the check that catches the original error. An RMSE of `0.06` on a
`[0, 1]` axis sounds excellent and means nothing: it cannot be compared against
another study, against a regulatory threshold, or against the previous hour's
reading. An RMSE of `19.2 µg/m³` can be.

The guard can be disabled with `require_physical_units=False`, but it has to be
disabled on purpose.

## 6. Baselines come first, and they are allowed to win

Four baselines are scored on exactly the same test windows as the networks:

| Baseline | Prediction | Trained? |
| --- | --- | :---: |
| persistence | the last observed value | no |
| seasonal naive | the value 24 hours earlier | no |
| window mean | the mean over the lookback window | no |
| ridge regression | a linear map of the flattened window | yes |

Persistence is the reference for skill scores. Skill of `0.0` means a model is
exactly as good as predicting no change; **negative skill means the model is
worse than doing nothing**, which is a result worth knowing and rarely reported.

Ridge is the honest middle ground. It sees identical inputs with no architecture
and no training curve. If a recurrent network cannot beat a linear map of the
same window, the sequence structure is not where the signal lives.

## 7. Every architecture gets the same budget

LSTM, GRU, CNN and CNN-LSTM are built with identical `units`, `filters`,
`dropout`, learning rate, batch size, patience and epoch limit. A comparison
where one architecture received a hyperparameter sweep and the others did not
measures the sweep.

`tests/test_deep.py` asserts the parameter counts stay within a factor of three
of each other, so a future edit cannot quietly hand one model more capacity.

### Convolutions are causal

`Conv1D` defaults to `padding="valid"`; `padding="same"` is the usual fix for
keeping the sequence length, and on a forecasting problem it is leakage — a
convolution centred on step *t* reads step *t+1*. Both convolutional models use
`padding="causal"`, and `tests/test_deep.py` verifies it empirically: perturbing
the final input timestep must leave every earlier activation unchanged.

### Early stopping restores the best weights

`restore_best_weights=True`. Without it the reported model is whatever the final
epoch produced, which on a series this size is frequently worse than something
several epochs earlier.

## 8. One run is not a result

Network training is stochastic. Two architectures separated by less than the
seed-to-seed spread are not distinguishable, and declaring a winner from a
single run is how noise becomes a finding.

`scripts/run_seed_sweep.py` repeats the whole comparison across seeds and
reports mean, standard deviation, minimum and maximum RMSE per model. Where the
gap between two models is smaller than that spread, `docs/results.md` says they
are indistinguishable rather than ranking them.

## What this pipeline does not do

- **No hyperparameter search.** Reported numbers come from one fixed
  configuration. A tuned model would likely do better; that experiment has not
  been run, so no claim is made about it.
- **No multi-site or multi-year validation.** The dataset is one roadside
  monitor in one Italian city over roughly one year. Nothing here supports a
  claim about how these models transfer.
- **No exogenous inputs.** No weather forecasts, no traffic counts, no calendar
  of public holidays. A production forecaster would use all three.
- **Not deployed.** This is a reproducible experiment, not a running service.
