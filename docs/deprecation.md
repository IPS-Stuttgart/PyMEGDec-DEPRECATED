# PyMEGDec deprecation shim

PyMEGDec is being phased out as an independent analysis package. During the
transition it remains available as a compatibility layer so old scripts and CI
jobs continue to run while the maintained implementation moves to NeuRepTrace.

The compatibility layer has two purposes:

1. keep existing `pymegdec ...` commands callable;
2. emit a visible warning with the intended NeuRepTrace replacement.

## Command mapping

| Legacy command | Intended replacement |
| --- | --- |
| `pymegdec cross-validate` | `neureptrace dataset run <dataset.yml> --analysis cross_validate` |
| `pymegdec transfer` | `neureptrace dataset run <dataset.yml> --analysis transfer_decode` |
| `pymegdec stimulus decoding` | `neureptrace dataset run <dataset.yml> --analysis stimulus_main_to_cue` |
| `pymegdec stimulus predictions` | `neureptrace dataset run <dataset.yml> --analysis stimulus_predictions` |
| `pymegdec stimulus temporal-generalization` | `neureptrace dataset run <dataset.yml> --analysis stimulus_temporal_generalization` |
| `pymegdec stimulus onset-scan` | `neureptrace dataset run <dataset.yml> --analysis stimulus_onset_scan` |
| `pymegdec stimulus robustness` | `neureptrace dataset run <dataset.yml> --analysis stimulus_robustness` |
| `pymegdec stimulus cross-subject-smoke` | `neureptrace dataset run <dataset.yml> --analysis cross_subject_smoke` |
| `pymegdec stimulus cross-subject-cue-calibrated` | `neureptrace dataset run <dataset.yml> --analysis cross_subject_cue_calibrated` |
| `pymegdec stimulus cross-subject-hyperalignment` | `neureptrace dataset run <dataset.yml> --analysis cross_subject_hyperalignment` |
| `pymegdec stimulus cross-subject-mcca` | `neureptrace dataset run <dataset.yml> --analysis cross_subject_mcca` |
| `pymegdec stimulus cross-subject-nested` | `neureptrace dataset run <dataset.yml> --analysis cross_subject_nested` |
| `pymegdec alpha metrics` | `neureptrace meg alpha-metrics <dataset.yml>` |
| `pymegdec alpha movement` | `neureptrace meg alpha-movement <dataset.yml>` |
| `pymegdec alpha movement-results` | `neureptrace meg alpha-movement-results <dataset.yml>` |
| `pymegdec alpha reaction-time` | `examples/bush_meg/run_alpha_reaction_time.py <dataset.yml>` |
| `pymegdec make-synthetic-data` | `neureptrace fieldtrip make-synthetic-data` or a test-fixture helper |

Some replacements are target commands and may only become runnable after the
corresponding NeuRepTrace migration step has landed. The warnings intentionally
document the intended destination even before every destination command exists.

## Removal criteria

Do not remove the compatibility wrappers until the following are true:

- NeuRepTrace loads FieldTrip MATLAB files directly.
- NeuRepTrace has dataset YAML support for the Bush MEG file conventions.
- Same-dataset cross-validation parity has been checked against PyMEGDec.
- Main-to-cue and cue-to-main transfer decoding parity has been checked.
- Probability-observation exports replace PyMEGDec prediction tables.
- Alpha workflows are either ported into NeuRepTrace or moved into documented
  Bush-MEG example scripts.
- CI covers the compatibility warnings and at least one migrated FieldTrip path.
