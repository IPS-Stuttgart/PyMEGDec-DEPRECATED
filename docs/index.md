# PyMEGDec documentation

PyMEGDec provides project-specific utilities for MEG decoding experiments. The
package focuses on workflows that need the repository's participant-file naming
conventions, MATLAB `.mat` loading, CTF sensor geometry, MEG preprocessing, and
paper-facing exports.

The usual workflow is:

1. Configure the local MEG data directory.
2. Choose a participant or participant range.
3. Run cross-validation, model transfer, stimulus decoding, alpha-metric export,
   sensor-level alpha-movement export, or alpha/reaction-time analysis.
4. Analyze the exported CSV tables and plots.

## Boundary with NeuRepTrace

PyMEGDec should keep code that is tied to this MEG dataset and its analysis
scripts:

- `Part*Data.mat` and `Part*CueData.mat` conventions.
- Dataset-specific MATLAB loading and trial metadata interpretation.
- CTF sensor-position handling for alpha topography and movement proxies.
- Stimulus-decoding defaults used by the project workflows.
- Compatibility wrappers for existing scripts.

Reusable decoding functionality should live in
[NeuRepTrace](https://github.com/IPS-Stuttgart/NeuRepTrace) and be imported here:

- Feature-matrix and MNE `Epochs` decoding helpers.
- Classifier calibration and prediction diagnostics.
- Generic confusion/per-class metrics.
- Temporal generalization, onset/state inference, and report aggregation.

## Repository layout

```text
src/pymegdec/              Package source code
  alpha_signal.py          Alpha-band filtering and phase extraction
  alpha_metrics.py         Per-trial alpha power and phase-gradient export
  alpha_movement.py        Sensor-level alpha movement trajectory export
  alpha_visualization.py   Alpha signal and phase-shift plotting helpers
  reaction_time_analysis.py Alpha/RT joins and association summaries
  stimulus_decoding.py     Time-resolved train-main / validate-cue decoding
  classifiers.py           Classifier registry and optional model backends
  preprocessing.py         Filtering, downsampling, window extraction
  model_transfer.py        Train-on-experiment / validate-on-cue evaluation
  cross_validation.py      Single-dataset cross-validation routine
scripts/                   Paper-facing and diagnostic exports
tests/                     Unit and data-dependent unittest suites
.github/workflows/         CI jobs for unit and data-dependent test subsets
```

Top-level files such as `cross_validation.py`, `evaluate_model_transfer.py`,
`export_alpha_metrics.py`, `analyze_stimulus_decoding.py`, and
`analyze_alpha_movement.py` are compatibility entry points for direct script use.
New code should prefer the package modules under `src/pymegdec/` and the grouped
`pymegdec` command where possible.
