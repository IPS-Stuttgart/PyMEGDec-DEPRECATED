# Archive policy

PyMEGDec is a legacy compatibility and reproducibility archive for the historical
BUSH-MEG decoding work. NeuRepTrace is the maintained successor for reusable
M/EEG decoding, dataset specifications, probability-observation diagnostics,
onset/state inference, calibration-aware metrics, and report aggregation.

## Maintained successor

Use [NeuRepTrace](https://github.com/IPS-Stuttgart/NeuRepTrace) for all new work.
New reusable code should not be added to PyMEGDec unless it is a temporary shim
that preserves an old command surface while delegating to NeuRepTrace.

## Supported residual changes

Accept changes here only when they are required for at least one of these cases:

- reproducing historical PyMEGDec/BUSH-MEG results;
- preserving old command names while routing users to NeuRepTrace;
- documenting the migration or archive boundary;
- keeping package metadata, tests, security notes, and CI usable for the archive;
- fixing a narrowly scoped regression in a historical workflow.

Do not add new datasets, new reusable decoders, new benchmark infrastructure,
new calibration metrics, or new report aggregation helpers here. Implement those
in NeuRepTrace and keep PyMEGDec as a caller or dataset-specific wrapper only
when backward compatibility requires it.

## Residual PyMEGDec ownership

The only code that should remain project-owned here is code that is tightly tied
to the historical repository context:

- `Part*Data.mat` and `Part*CueData.mat` naming conventions;
- BUSH-MEG metadata defaults and private result-layout compatibility;
- CTF geometry handling used by the legacy alpha analyses;
- alpha-band, alpha-movement, and alpha/reaction-time paper exports;
- exact historical CSV schemas required to reproduce old outputs;
- thin compatibility entry points for old scripts and notebooks.

## Final closeout checklist

Before archiving the GitHub repository, verify the following manually:

1. The README and documentation point users to NeuRepTrace.
2. `pyproject.toml` marks the package as inactive/deprecated.
3. Public imports and command-line entry points are kept only for compatibility.
4. Fast tests pass without private MEG files.
5. Any remaining private-data-dependent commands document their required input
   files and are skipped gracefully when those files are absent.
6. Open issues and pull requests are either closed, migrated to NeuRepTrace, or
   explicitly labelled as archive-only.
7. Repository settings are changed to archived/read-only after the final closeout
   PR is merged.

## Security and support

This archive is not actively maintained for feature development. Security fixes
should be limited to dependency metadata or compatibility changes needed to keep
historical reproduction environments installable. General usage questions and
new feature requests should be opened in NeuRepTrace instead.
