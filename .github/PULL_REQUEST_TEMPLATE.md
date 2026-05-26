## Archive-scope checklist

PyMEGDec is deprecated and kept only as a historical compatibility archive. Before merging, confirm that this PR is archive-scoped.

- [ ] This change is required for historical PyMEGDec/BUSH-MEG reproducibility, archive hygiene, or a temporary NeuRepTrace compatibility shim.
- [ ] No new reusable dataset loader, decoder, benchmark, calibration metric, report helper, or analysis workflow is being added here.
- [ ] Any reusable functionality has been implemented in or redirected to NeuRepTrace.
- [ ] Fast tests pass without private MEG files, or the reason they were not run is documented below.
- [ ] Private-data-dependent behavior is skipped gracefully when the historical MAT files are unavailable.

## Summary


## Validation

