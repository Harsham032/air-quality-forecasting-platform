# Results

Every number in this document was produced by running the code in this
repository. Nothing is estimated, rounded up, or carried over from elsewhere.
Where a result is unfavourable to the deep learning models, it is reported as
measured.

## Environment

| | |
| --- | --- |
| Python | 3.11.15 |
| Platform | Linux 6.18 x86-64, glibc 2.39 |
| CPU | 4 cores, no GPU |
| Memory | 15 GB |
| NumPy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| TensorFlow | 2.21.0 (CPU) |

Reproduce with `make experiment` (about 2.5 minutes on this machine) and
`make sweep` (about 13 minutes). Every run writes its own environment block to
`reports/<name>-summary.json`.

## Dataset

UCI Air Quality, dataset 360. One roadside gas multisensor in an Italian city.

| | |
| --- | --- |
| Rows as read | 9,357 |
| Columns as read | 15 (13 numeric) |
| Cells holding the `-200` sentinel | **16,701** |
| Rows containing at least one sentinel | **8,530 (91.2%)** |
| Columns dropped (observed < 50%) | `NMHC(GT)` |
| Rows dropped to gaps longer than 6 hours | 1,737 |
| Cells filled by bounded interpolation | 1,693 |
| **Rows retained** | **7,620** |
| **Columns retained** | **12** |
| Retained period | 2004-03-10 18:00 to 2005-04-04 14:00 |

### What the sentinel does to the summary statistics

| Column | Observed hours | Sentinel cells | Observed share | Mean if `-200` is data | Mean over observed hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CO(GT)` | 7,674 | 1,683 | 82.0% | **−34.21** | 2.15 |
| `PT08.S1(CO)` | 8,991 | 366 | 96.1% | 1048.99 | 1099.83 |
| `NMHC(GT)` | 914 | 8,443 | **9.8%** | −159.09 | 218.81 |
| `C6H6(GT)` | 8,991 | 366 | 96.1% | 1.87 | 10.08 |
| `PT08.S2(NMHC)` | 8,991 | 366 | 96.1% | 894.60 | 939.15 |
| `NOx(GT)` | 7,718 | 1,639 | 82.5% | 168.62 | 246.90 |
| `PT08.S3(NOx)` | 8,991 | 366 | 96.1% | 794.99 | 835.49 |
| **`NO2(GT)`** (target) | 7,715 | 1,642 | 82.5% | **58.15** | **113.09** |
| `PT08.S4(NO2)` | 8,991 | 366 | 96.1% | 1391.48 | 1456.26 |
| `PT08.S5(O3)` | 8,991 | 366 | 96.1% | 975.07 | 1022.91 |
| `T` | 8,991 | 366 | 96.1% | 9.78 | 18.32 |
| `RH` | 8,991 | 366 | 96.1% | 39.49 | 49.23 |
| `AH` | 8,991 | 366 | 96.1% | **−6.84** | 1.03 |

Two columns average to a negative number if the sentinel is read as a
measurement. `NMHC(GT)` is observed in under 10% of hours, with a single
unbroken gap of **8,126 hours (339 days)**; it is dropped rather than imputed.

## The correction, measured directly

`examples/sentinel_impact.py` runs two pipelines on the same file with the same
architecture (GRU, 32 units), the same seed, the same 8 epochs, and the same
chronological split. The only difference is whether `-200` is treated as a
measurement.

| | Naive pipeline | Corrected pipeline |
| --- | ---: | ---: |
| `NO2(GT)` mean | 58.15 µg/m³ | 110.71 µg/m³ |

| Scaler range | `[-200, 340]` | fitted on observed hours only |
| Share of the `[0,1]` axis holding real data | **62.6%** | 100% |
| RMSE on the scaled axis | **0.1280** | — (refused) |
| **R²** | **0.1592** | **0.8231** |
| **RMSE in physical units** | **69.12 µg/m³** | **20.86 µg/m³** |

(The corrected mean of 110.71 is over the 7,620 *retained* hours; the 113.09 in
the table above is over all observed hours in the file, before rows spanning
long gaps are dropped. Both are correct for what they describe.)

The naive pipeline's error is **3.3× larger** in physical units. Its reported
RMSE of `0.1280` gives no hint of that: it is a small number on an axis that
means nothing, and no amount of staring at it reveals the problem.

This is the mechanism behind the near-zero and negative R² values that this
dataset produces in uncorrected pipelines. The model is not underfitting; it is
being trained on a distribution where 37% of the range is occupied by a spike of
missing-value codes.

## Main comparison

Target `NO2(GT)`, 24-hour lookback, 1-hour horizon, chronological 70/15/15
split, 1,119 test hours, seed 20260101. Configuration: `configs/default.yaml`.

| Model | Kind | RMSE (µg/m³) | MAE (µg/m³) | R² | Bias (µg/m³) | Skill vs persistence | Fit (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **ridge regression** | baseline | **18.603** | 13.734 | 0.8593 | −1.275 | **0.2381** | — |
| gru | deep | 19.245 | 14.597 | 0.8495 | 1.946 | 0.2118 | 48.5 |
| lstm | deep | 19.799 | 14.808 | 0.8407 | 1.387 | 0.1891 | 33.2 |
| cnn_lstm | deep | 20.122 | 15.250 | 0.8354 | 0.087 | 0.1759 | 39.2 |
| persistence | baseline | 24.418 | 18.341 | 0.7577 | 0.029 | 0.0000 | — |
| cnn | deep | 33.052 | 25.424 | 0.5560 | −5.304 | **−0.3536** | 11.8 |
| seasonal naive (24h) | baseline | 40.260 | 30.786 | 0.3412 | 0.988 | −0.6488 | — |
| window mean | baseline | 44.960 | 36.113 | 0.1784 | 0.500 | −0.8413 | — |

**No neural network beat ridge regression.**

## Seed sweep

Five seeds (20260101–20260105), same configuration. The deterministic baselines
repeat exactly, which is why their standard deviation is zero.

| Model | Kind | Mean RMSE | SD | Min | Max | Mean R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ridge regression | baseline | **18.603** | 0.000 | 18.603 | 18.603 | 0.8593 |
| gru | deep | 19.060 | 0.160 | 18.859 | 19.245 | 0.8523 |
| cnn_lstm | deep | 19.680 | 0.471 | 19.160 | 20.122 | 0.8425 |
| lstm | deep | 19.705 | 0.431 | 19.105 | 20.269 | 0.8421 |
| persistence | baseline | 24.418 | 0.000 | 24.418 | 24.418 | 0.7577 |
| cnn | deep | 32.863 | 0.662 | 32.174 | 33.689 | 0.5609 |
| seasonal naive (24h) | baseline | 40.260 | 0.000 | 40.260 | 40.260 | 0.3412 |
| window mean | baseline | 44.960 | 0.000 | 44.960 | 44.960 | 0.1784 |

Per-seed RMSE:

| Seed | ridge | gru | lstm | cnn_lstm | cnn | persistence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260101 | 18.603 | 19.245 | 19.799 | 20.122 | 33.052 | 24.418 |
| 20260102 | 18.603 | 18.967 | 19.844 | 19.160 | 33.201 | 24.418 |
| 20260103 | 18.603 | 18.859 | 19.105 | 19.801 | 33.689 | 24.418 |
| 20260104 | 18.603 | 19.035 | 19.508 | 19.208 | 32.174 | 24.418 |
| 20260105 | 18.603 | 19.193 | 20.269 | 20.111 | 32.197 | 24.418 |

**Ridge regression is ahead of the best network on 5 of 5 seeds.** The GRU's own
seed-to-seed standard deviation is 0.160 µg/m³ and its best single run (18.859)
still does not reach ridge's 18.603. The ordering is consistent, not noise.

## Interpretation

**The networks work. They just do not win here.** Every architecture except the
pure CNN reaches R² above 0.84 and removes 18–21% of persistence's error. That
is a functioning forecaster, and it is the opposite of what the uncorrected
pipeline produces on the same data.

**A linear model on the same window does slightly better.** Ridge sees exactly
the same 288 inputs (24 hours × 12 features) with no architecture, no training
curve, and no fit time worth measuring. At a one-hour horizon, hourly NO₂ is
close to linear in its recent history, and there is not much non-linear structure
for a recurrent model to find. This is a real finding about the problem, not a
failure of the implementation.

**Persistence is a serious opponent.** At a one-step-ahead horizon the previous
hour explains 76% of the variance by itself. Every model in the table has to be
read against that, and two of the four "naive" baselines are worse than useless.

**The CNN is genuinely bad, and it is informative that it is.** Two causal
convolutions followed by global average pooling deliberately discard position:
the pooled representation cannot tell whether a spike occurred one hour ago or
twenty. At this horizon that is exactly the information that matters, and its
skill score of −0.354 says so — it is worse than predicting no change at all. The
CNN-LSTM, which keeps the recurrent layer, does not have this problem.

**Bias is worth reading next to RMSE.** The GRU runs +1.95 µg/m³ high and the CNN
−5.30 µg/m³ low. Persistence is almost unbiased (+0.03) by construction. A model
with low RMSE and high bias is systematically wrong in one direction, which
matters for a threshold-exceedance application in a way RMSE alone does not show.

**What would probably change the answer.** A longer horizon. Persistence decays
quickly past a few hours while a sequence model's advantage should grow;
`configs/horizon-6h.yaml` is set up for that experiment. It has not been run, so
nothing is claimed about it here.

## Limitations

These bound every number above.

- **One site, one year, one target.** A single roadside monitor in one city from
  March 2004 to April 2005. There is one winter, so any seasonal effect is
  observed once and cannot be separated from that year's weather. Nothing here
  supports a claim about other sites, other pollutants, or other periods.
- **No hyperparameter search.** All numbers come from one fixed configuration
  chosen before any results were seen. A tuned network would likely close the
  0.46 µg/m³ gap to ridge, and possibly cross it. That experiment has not been
  run.
- **18.6% of rows were dropped, non-randomly.** The long gaps correspond to
  analyser downtime, which is not independent of conditions. Dropping them is the
  honest option available, not an unbiased one.
- **1,693 retained cells are interpolated, not measured.** Roughly 1.9% of the
  retained data is reconstructed.
- **Sensor drift is present.** The metal-oxide responses drift across the
  deployment, so a model trained on early months and tested on later ones is
  partly learning to track that drift. The chronological split makes this a real
  difficulty rather than hiding it, but it is not corrected for.
- **One-hour horizon only.** Results at other horizons are not measured.
- **Five seeds.** Enough to show the ordering is consistent; not enough to put a
  confidence interval on the gap.
- **Not deployed.** This is a reproducible experiment. There is no service, no
  users, no production traffic, and no claim of any.

## Next experiments

In the order most likely to change a conclusion:

1. **Longer horizons** (6, 12, 24 hours). The one experiment most likely to
   reverse the headline result, since persistence and ridge both weaken quickly
   as the horizon grows.
2. **A tuned network against a tuned ridge.** A fair sweep means sweeping both.
3. **Multi-step output** instead of repeated one-step forecasts, which compounds
   error.
4. **Calendar features** — hour of day and day of week as explicit inputs rather
   than something the model must infer from 24 lags.
5. **Quantile or interval forecasts.** For an air quality application, the
   probability of exceeding a threshold is more useful than a point estimate.
6. **A second site,** to find out whether any of this transfers. Until that is
   run, it should be assumed it does not.
