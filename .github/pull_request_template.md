## Archive-maintenance reason

PyMEGDec is deprecated and in archive-closure mode. Use NeuRepTrace for new reusable work.

Please check the applicable reason for changing this repository:

- [ ] Security, packaging, dependency, or CI fix needed to keep legacy workflows installable.
- [ ] Documentation-only clarification for reproducing historical outputs.
- [ ] Thin compatibility shim that forwards an old PyMEGDec entry point to NeuRepTrace.
- [ ] Removal or cleanup after the corresponding NeuRepTrace workflow has been validated.
- [ ] Other archive-maintenance reason, explained below.

## Scope boundary

- [ ] This PR does not add new reusable decoding, dataset-loading, evaluation, diagnostic, benchmark, or reporting functionality to PyMEGDec.
- [ ] Any reusable scientific functionality has been implemented in, or intentionally deferred to, NeuRepTrace.
- [ ] Remaining PyMEGDec-specific behavior is tied to historical BUSH-MEG reproduction, old file naming conventions, or paper-specific exports.

## Validation

- [ ] `python -m unittest discover -v` passes or the failure is explained below.
- [ ] Dataset-spec or NeuRepTrace migration commands were checked when relevant.
- [ ] Data-dependent reproduction commands are documented when private MEG files are required.

## Notes

