# Data

Nothing in this directory is committed. Acquire it with:

```bash
python scripts/download_data.py     # or: make data
```

| | |
| --- | --- |
| `raw/` | the downloaded file, exactly as published |
| `interim/` | intermediate artefacts from ad-hoc work |
| `processed/` | cleaned frames, if cached |

The source dataset is UCI Air Quality (dataset 360), distributed under CC BY
4.0. It is downloaded rather than redistributed here. See
[../docs/data-sources.md](../docs/data-sources.md) for its contents, licence,
and the three file-format traps that change the numbers if handled wrong.

The test suite does not require this directory to be populated: tests that need
the real file are marked `slow` and skip when it is absent.
