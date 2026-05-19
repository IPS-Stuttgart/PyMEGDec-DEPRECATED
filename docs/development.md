# Development

## Test strategy

Run the full default suite from the repository root:

```bash
python -m unittest discover -v
```

The suite includes fast tests that do not require private MEG files. Tests that
need participant MAT files should skip cleanly when the data directory cannot be
resolved.

To run data-dependent tests, configure the data directory first:

```bash
export PYMEGDEC_DATA_DIR=/path/to/MEG-Data
python -m unittest discover -v
```

On PowerShell:

```powershell
$env:PYMEGDEC_DATA_DIR = "C:\path\to\MEG-Data"
python -m unittest discover -v
```

## Static checks

The project configuration includes Ruff, mypy, Black, isort, and pylint settings
in `pyproject.toml`. Use the GitHub Actions workflows as the source of truth for
which checks are required before merging.

## Documentation maintenance

Documentation is source-controlled Markdown in `docs/` and is built with
MkDocs. Keep the README short and move detailed workflow instructions into the
relevant documentation page.

Recommended structure for future additions:

- New user workflow: add or update a page under `docs/` and link it from
  `mkdocs.yml`.
- New command-line option: document it in `docs/cli.md` or the workflow-specific
  page.
- New output table: document the producer command, row granularity, and the most
  important columns near the workflow that creates it.
- New reusable decoding method: consider whether it belongs in NeuRepTrace instead
  of PyMEGDec.

Preview locally:

```bash
python -m pip install mkdocs
mkdocs serve
```

Validate before committing:

```bash
mkdocs build --strict
```

## Output and provenance guidelines

When adding new exports, prefer explicit output paths and deterministic rows.
For paper-facing workflows, include enough metadata columns to reconstruct the
run, for example participant id, train/validation direction, window center,
classifier, PCA setting, frequency range, and random seed where applicable.
