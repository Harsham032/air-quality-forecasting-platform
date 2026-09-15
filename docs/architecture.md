# Architecture

## Shape of the code

```
src/aqf/
  config.py              experiment config (YAML) and deployment settings (env)
  errors.py              one exception hierarchy, so failures are catchable
  logging_utils.py       structured logging
  experiment.py          the run: load, split, fit, score, save
  data/
    loader.py            sentinel handling, missingness reporting, cleaning
    windows.py           chronological split, train-only scaling, windowing
  models/
    baselines.py         persistence, seasonal naive, window mean, ridge
    deep.py              LSTM, GRU, CNN, CNN-LSTM
  evaluation/
    metrics.py           RMSE/MAE/MAPE/R2/bias in physical units, skill scores
  reporting/
    tables.py            markdown tables
    plots.py             figures
scripts/
  download_data.py       fetch the dataset
  run_experiment.py      one comparison, one config
  run_seed_sweep.py      the same comparison across seeds
  secrets_scan.py        pre-push credential check
```

## Data flow

```
data/raw/AirQuality.csv
        |
        |  load_air_quality
        |    - parse with an explicit timestamp format
        |    - replace -200 with NaN            <-- the correction
        |    - drop columns below the observed-share threshold
        |    - interpolate short gaps, drop rows spanning long ones
        v
  cleaned frame (7,620 hours x 12 columns) + DatasetReport
        |
        |  split_by_time
        |    - divide chronologically 70/15/15
        |    - fit mean and scale on TRAIN ONLY
        |    - window within each part, never across a boundary
        v
  TemporalSplit (X, y per part, scaled; target_mean/target_scale retained)
        |
        +-- baselines ------> predictions (scaled)
        |
        +-- networks -------> predictions (scaled)
                                  |
                                  |  split.inverse_target
                                  v
                          predictions in ug/m3
                                  |
                                  |  evaluate_forecast
                                  |    - refuses a target still on a [0,1] axis
                                  v
                          ForecastMetrics + skill_score
                                  |
                                  v
                   reports/*.csv, reports/*.json, figures/*.png
```

## Design decisions

### The correction lives in one place

`split.inverse_target` is called in `_score`, which every model — baseline and
network alike — passes through. There is no code path that scores a scaled
prediction, and `evaluate_forecast` raises if one is ever constructed.

Putting the inversion in each model's scoring block instead would work until
someone adds a fifth model and forgets.

### Reports describe the file *and* what survived

`DatasetReport` carries both: `n_rows` is what was read, `n_rows_retained` is
what came out, and `interpolated_cells` counts how much of the retained series
is reconstructed rather than measured. Conflating those is how a report ends up
claiming 9,357 clean hours from a file that yields 7,620.

### `ColumnReport.usable` stores its own threshold

A boolean that depends on a configurable threshold has to carry that threshold,
or it silently lies when the threshold is raised. It is a field, not a constant.

### The orchestrator is a library function, not a script

`run_experiment(config, frame=...)` accepts a pre-loaded frame, so the test
suite exercises the exact code path a real run takes, on synthetic data, in
seconds. `scripts/run_experiment.py` is argument parsing and printing.

### TensorFlow is imported lazily

`models/deep.py` imports TensorFlow inside its functions. Importing `aqf.config`
or `aqf.data` therefore costs nothing, which keeps the fast test suite fast and
lets the data-handling half of the project be used without TensorFlow installed.

### Errors are typed

Everything raises a subclass of `AqfError`: `ConfigurationError`, `DataError`,
`DataQualityError`, `ModelError`, `EvaluationError`. A caller can distinguish
"your YAML is wrong" from "this column is 90% missing" without matching on
message text, and the scripts turn them into an exit code rather than a
traceback.

## Reproducibility

- Seeds are set through `tf.keras.utils.set_random_seed` and `np.random.seed`
  from `run.seed` in the config.
- Every run writes `reports/<name>-summary.json` containing the full resolved
  config, the library versions it ran against, the split sizes, and every
  outcome. A result that cannot be traced to its configuration is not a result.
- Results in `docs/results.md` state the environment they were measured on.

Exact bit-for-bit reproducibility across machines is not claimed: CPU thread
scheduling and oneDNN kernel selection make floating-point reductions
order-dependent. The seed sweep is what establishes that a conclusion survives
that variation.
