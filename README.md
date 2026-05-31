# PyMEGDec (deprecated)

> [!WARNING]
> **PyMEGDec is deprecated. Use [NeuRepTrace](https://github.com/IPS-Stuttgart/NeuRepTrace) for all new work from now on.** This repository is kept only as a compatibility and reproducibility archive for the historical MEG decoding workflows. Do not add new reusable dataset-loading, decoding, evaluation, diagnostics, or reporting functionality here; implement it in NeuRepTrace instead.

## Archive status

PyMEGDec is in final archive mode. The maintained successor is
[NeuRepTrace](https://github.com/IPS-Stuttgart/NeuRepTrace), which owns reusable
M/EEG decoding, dataset-spec validation, probability-observation diagnostics,
onset/state inference, calibration-aware metrics, and report aggregation.

This repository should receive only narrowly scoped closeout changes:

- fixes that keep historical PyMEGDec/BUSH-MEG reproduction commands runnable;
- documentation, security, packaging, or CI changes needed for archive hygiene;
- temporary compatibility shims that route users to NeuRepTrace.

New reusable methods, dataset loaders, benchmarks, or analysis workflows belong
in NeuRepTrace. See [`ARCHIVE.md`](ARCHIVE.md) for the archive policy and the
final maintainer checklist.

PyMEGDec contains the MEG-specific analysis layer for historical decoding
experiments. It loads participant MATLAB files, prepares MEG windows, runs
model-transfer and cross-validation workflows, and exports stimulus analysis
tables for reproducibility of the legacy project results.

Generic decoding summaries and reusable prediction-table diagnostics belong in
[NeuRepTrace](https://github.com/IPS-Stuttgart/NeuRepTrace). PyMEGDec now has a
legacy compatibility role: dataset file conventions and metadata mappings can be
expressed as NeuRepTrace YAML/JSON dataset specs, while highly project-specific
alpha, CTF geometry, reaction-time, and paper-export scripts remain here only
until they are no longer needed for reproducing old runs.

Write a starter NeuRepTrace dataset spec for the historical `Part*Data.mat` /
`Part*CueData.mat` convention with:

```bash
pymegdec data write-neureptrace-spec --out configs/bushmeg.yml
neureptrace dataset validate configs/bushmeg.yml
```

The alpha-band, alpha-movement, and alpha/reaction-time workflows are now
explicitly legacy-only. They remain callable for reproducibility and to regenerate
existing Bush/MEG CSV exports, but new reusable decoding or dataset-loading work
must be implemented in NeuRepTrace instead.

## Legacy reproduction quick start

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
pymegdec stimulus cross-subject-smoke --data-dir /path/to/MEG-Data --participants 1-4,6,8,9,10,13-27
```

## NeuRepTrace dataset-spec migration

The root `dataset.yml` captures the current PyMEGDec participant-file
conventions as a NeuRepTrace dataset spec. This is the migration path for
turning PyMEGDec from an installable MEG-specific package into a study
configuration plus reproduction scripts.

```bash
export PYMEGDEC_DATA_DIR=/path/to/MEG-Data
neureptrace dataset validate dataset.yml
neureptrace dataset list-files dataset.yml
```

Keep MATLAB parsing, feature extraction, CTF geometry handling, and compatibility
shims in Python loaders. Keep paths, participant ranges, file-role mappings,
default windows, and output locations in the dataset spec.

## Documentation

The longer workflow documentation lives in `docs/`:

- `docs/archive.md` — archive policy, supported residual changes, and closeout checklist.
- `docs/getting-started.md` — installation, optional extras, and tests.
- `docs/data.md` — data-directory resolution and participant-file conventions.
- `docs/cli.md` — grouped CLI commands and compatibility entry points.
- `docs/stimulus-decoding.md` — time-resolved stimulus decoding, diagnostics,
  robustness exports, temporal generalization, and onset scanning.
- `docs/alpha.md` — legacy alpha metrics, sensor-level alpha movement, and
  alpha/RT analysis kept for reproducibility during the PyMEGDec phase-out.
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
