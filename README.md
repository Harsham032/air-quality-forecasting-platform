# Air Quality Forecasting Platform

Hourly NO₂ forecasting on the UCI Air Quality dataset, built around a specific
question: **does a deep sequence model actually beat a simple baseline here?**

The pipeline exists because the usual answer to that question is unreliable for
two reasons that have nothing to do with the architecture.

## The two problems this project is built to avoid

**1. The dataset encodes missing readings as `-200`.** Not as blanks — as a
number, in a numeric column, that every library will average and scale without
complaint. It appears in 16,701 cells across 91.2% of the rows.

Read naively, the carbon monoxide column has a mean of **−34.21**. A negative
concentration is not a small inaccuracy; it is impossible, and it is what the
arithmetic gives you.

**2. Min-max scaling then hides the damage.** With `-200` present, `NO2(GT)`
scales over the range `[-200, 333]` instead of `[2, 333]`. The real
measurements are compressed into a narrow band at the top of the axis while
every missing hour sits at exactly zero. The model is being asked to predict a
spike train, it cannot, and its error is reported on the same distorted axis —
so nothing in the output announces the problem.

This repository fixes both, and makes it structurally difficult to reintroduce
either:

- the sentinel is converted to `NaN` before anything else touches the frame, and
  both means are printed side by side so the difference is visible;
- `evaluate_forecast` **raises** if asked to score a target that still looks
  min-max scaled. Errors are reported in µg/m³ or not at all.

## What it does

| | |
| --- | --- |
| **Target** | `NO2(GT)`, hourly, µg/m³ |
| **Data** | UCI Air Quality, 9,357 hours → 7,620 after cleaning |
| **Split** | chronological 70/15/15, scaler fitted on training only |
| **Baselines** | persistence, seasonal naive (24h), window mean, ridge |
| **Networks** | LSTM, GRU, CNN, CNN-LSTM — identical capacity and budget |
| **Reported** | RMSE, MAE, R², bias in µg/m³, plus skill against persistence |

The baselines are not a formality. They are scored on identical windows, and
they are allowed to win — see [docs/results.md](docs/results.md) for what
actually happened.

## Quickstart

```bash
make install          # venv, dependencies, editable install
make data             # download the UCI dataset into data/raw
make smoke            # prove the pipeline runs (3 epochs; not a result)
make experiment       # the full comparison, ~10 minutes on a laptop CPU
make sweep            # repeat across 5 seeds and report the spread
```

Or directly:

```bash
python scripts/run_experiment.py --config configs/default.yaml
python scripts/run_experiment.py --config configs/default.yaml --set split.horizon=6
```

Each run writes to `reports/`: a leaderboard, the test-period predictions, the
data-quality table, and a JSON summary carrying the full resolved config and the
library versions it ran against. Figures go to `figures/`.

## Reading the numbers

**Skill score** is the fraction of persistence's RMSE a model removes. `0.0`
means exactly as good as predicting no change. **Negative skill means the model
is worse than doing nothing** — a result worth reporting and rarely reported.

**R² below zero** means a model is beaten by a constant. On a series with a
daily cycle this obvious, a negative R² is not a weak model; it is a signal that
something upstream is broken. It is precisely what the uncorrected pipeline
produces.

**Errors are in µg/m³.** An RMSE of `0.06` on a `[0, 1]` axis cannot be compared
against another study, against a regulatory threshold, or against last hour's
reading. An RMSE of `19.2 µg/m³` can be.

## Configuration

Experiment settings live in YAML and are overridable per run:

```bash
python scripts/run_experiment.py --set models.epochs=10 --set models.architectures='[gru, lstm]'
```

Unknown keys are rejected rather than ignored, so a typo fails immediately
instead of silently running the default. Deployment settings come from `AQF_*`
environment variables; see `.env.example`. Credentials are never read from YAML.

## Development

```bash
make check       # ruff, black, mypy, and the fast test suite
make test-all    # including the tests that train networks
make coverage
```

The test suite asserts the things that matter and are easy to get wrong:

- the sentinel never survives loading, and a naive read produces a lower mean
- a stricter drop threshold actually drops more, and `usable` agrees with it
- long gaps are dropped rather than bridged, and interpolated cells are counted
- **doubling every value after the training cut does not move the training
  scaler by a single float** — the direct test for scaler leakage
- no window straddles a split boundary, asserted by exact count
- convolutions are causal: perturbing the last input timestep leaves every
  earlier activation unchanged
- `evaluate_forecast` refuses a `[0, 1]` target
- the printed verdict cannot claim a win the leaderboard does not show

## Documentation

| | |
| --- | --- |
| [docs/results.md](docs/results.md) | measured results, with limitations |
| [docs/methodology.md](docs/methodology.md) | every choice, and the error it prevents |
| [docs/data-sources.md](docs/data-sources.md) | the dataset, its licence, its traps |
| [docs/architecture.md](docs/architecture.md) | code layout and data flow |

## Scope

One roadside monitor, one Italian city, roughly one year, one target. No
hyperparameter search was run. Nothing here is deployed, and no claim is made
about production use, multi-site transfer, or performance beyond what
`docs/results.md` reports measuring.

## Licence

MIT. The dataset is distributed by UCI under CC BY 4.0 and is not redistributed
here — `scripts/download_data.py` fetches it.
