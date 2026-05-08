# Getting started

## Requirements

PyMEGDec targets Python `>=3.11,<3.14`. The project uses Poetry for local
development and dependency resolution.

## Install for development

From the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install poetry
poetry install
```

Run commands inside the Poetry environment:

```bash
poetry run pymegdec --help
poetry run python -m unittest discover -v
```

## Optional classifier backends

The default install includes the core scientific stack and the RepTrace-backed
classifier registry. Additional classifier backends are available through extras:

```bash
poetry install --extras "xgboost"
poetry install --extras "torch"
poetry install --extras "all"
```

The PyMEGDec-specific classifier additions are:

- `xgboost`, which requires the `xgboost` extra.
- `pytorch-mlp`, which requires the `torch` extra.

Other classifier names are inherited from RepTrace's classifier registry.

## Smoke tests

Run the default test suite:

```bash
python -m unittest discover -v
```

Fast unit tests are designed to run without private MEG files. Data-dependent
integration tests are skipped when the data directory cannot be resolved.

Create a private-data-free demo directory when you want to exercise the command-line workflows without access to the real MEG files:

```bash
pymegdec make-synthetic-data --out demo-data
pymegdec cross-validate --data-dir demo-data --participant 2
pymegdec transfer --data-dir demo-data --participant 2 --null-window-center nan
```

To run the integration checks, configure a data directory containing files such
as `Part2Data.mat` and `Part2CueData.mat` before invoking the same test command.

## Documentation preview

The documentation is plain Markdown under `docs/` and can be served with MkDocs:

```bash
python -m pip install mkdocs
mkdocs serve
```

Build the static site locally with strict link checking:

```bash
mkdocs build --strict
```
