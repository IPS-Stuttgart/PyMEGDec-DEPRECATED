# PyMEGDec

PyMEGDec contains the MEG-specific analysis layer for decoding experiments. It
loads participant MATLAB files, prepares MEG windows, runs model-transfer and
cross-validation workflows, and exports stimulus, alpha, and reaction-time
analysis tables.

Generic decoding summaries and reusable prediction-table diagnostics belong in
[RepTrace](https://github.com/IPS-Stuttgart/RepTrace). PyMEGDec keeps the
project-specific data conventions, preprocessing defaults, CTF sensor-geometry
handling, and paper-facing scripts.

## Quick start

```bash
python -m pip install --upgrade pip
python -m pip install poetry
poetry install
```

Install optional classifier backends when needed:

```bash
poetry install --extras "all"
```

Configure the data directory with `--data-dir`, `PYMEGDEC_DATA_DIR`, or an
ignored `.pymegdec-data-dir` file. Participant files are expected to follow the
`Part2Data.mat` and `Part2CueData.mat` naming convention.

```bash
pymegdec cross-validate --data-dir /path/to/MEG-Data --participant 2
pymegdec transfer --data-dir /path/to/MEG-Data --participant 2 --null-window-center nan
pymegdec stimulus-decoding --data-dir /path/to/MEG-Data --participants 2 --output outputs/part2_stimulus_decoding.csv
```

## Documentation

The longer workflow documentation lives in `docs/`:

- `docs/getting-started.md` — installation, optional extras, and tests.
- `docs/data.md` — data-directory resolution and participant-file conventions.
- `docs/cli.md` — grouped CLI commands and compatibility entry points.
- `docs/stimulus-decoding.md` — time-resolved stimulus decoding, diagnostics,
  robustness exports, temporal generalization, and onset scanning.
- `docs/alpha.md` — alpha metrics, sensor-level alpha movement, and alpha/RT
  analysis.
- `docs/api.md` — public Python entry points and module boundaries.
- `docs/development.md` — test strategy and documentation maintenance.

To preview the documentation locally:

```bash
python -m pip install mkdocs
mkdocs serve
```

## Tests

```bash
python -m unittest discover -v
```

Fast tests run without private MEG files. Data-dependent tests are skipped when
the participant MAT files cannot be resolved.
