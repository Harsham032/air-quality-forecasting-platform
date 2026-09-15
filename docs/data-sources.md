# Data sources

## UCI Air Quality

- **Source**: UCI Machine Learning Repository, dataset 360
- **Landing page**: https://archive.ics.uci.edu/dataset/360/air+quality
- **Reference**: De Vito, S., Massera, E., Piga, M., Martinotto, L., & Di
  Francia, G. (2008). On field calibration of an electronic nose for benzene
  estimation in an urban pollution monitoring scenario. *Sensors and Actuators
  B: Chemical*, 129(2), 750–757.
- **Licence**: CC BY 4.0, per the repository's dataset page.
- **Acquisition**: `python scripts/download_data.py`, which fetches the archive
  and extracts the CSV to `data/raw/AirQuality.csv`. The file is not committed;
  `data/` is gitignored.

### What it contains

9,357 hourly records from a gas multisensor device deployed at road level in an
Italian city, from March 2004 to April 2005. Each row carries five metal-oxide
sensor responses (`PT08.S1` through `PT08.S5`), four co-located reference
analyser readings (`CO(GT)`, `NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`), and
temperature, relative humidity and absolute humidity.

The `(GT)` columns are the ground-truth reference measurements. `NO2(GT)` is the
forecasting target used throughout, in micrograms per cubic metre.

### File format traps

Three, all of which change the numbers if handled wrong:

1. **Separators.** Semicolon-delimited with a comma decimal mark. Read with
   default settings, every numeric column arrives as a string.
2. **Timestamps.** `Date` and `Time` are separate columns, and the time is
   written `18.00.00` with dots. Left to infer the format, pandas parses this
   inconsistently across rows, which silently reorders an hourly series. The
   loader passes `format="%d/%m/%Y %H.%M.%S"` explicitly.
3. **Trailing rows.** The exporter appends blank rows and two unnamed empty
   columns. Both are dropped on read.

### The missing-value sentinel

Missing readings are encoded as `-200`, not as blanks. See
[methodology.md](methodology.md) for what this does to an unprepared pipeline
and how the loader handles it. This is the single most important thing to know
about the dataset.

### Known limitations of the source data

- **One site.** A single roadside monitor. Concentrations at road level differ
  substantially from background urban levels, and nothing here generalises to
  other sites.
- **Just over one year.** 2004-03 to 2005-04. There is one winter and one
  summer, so any seasonal pattern is observed once and cannot be separated from
  that year's particular weather.
- **Sensor drift.** The reference paper is about calibrating these sensors; the
  metal-oxide responses drift over the deployment. A model trained on the early
  months and tested on the later ones is partly learning to track that drift.
  The chronological split makes this a genuine difficulty rather than hiding it.
- **Missingness is not random.** The long gaps correspond to analyser downtime,
  which is not independent of conditions. Dropping those rows is the honest
  option available, not an unbiased one.

## Data not used

The work this project rebuilds also drew on a merged air quality index dataset
covering a different region. That file is not part of this repository and no
result here depends on it; the scope is the UCI series alone. Any conclusion
below applies to that dataset and no other.

## Directory layout

```
data/
  raw/        the downloaded file, exactly as published
  interim/    intermediate artefacts from ad-hoc work
  processed/  cleaned frames, if cached
```

All three are gitignored. Nothing in `data/` is required to be present for the
test suite to pass; tests that need the real file are marked `slow` and skip
when it is absent.
