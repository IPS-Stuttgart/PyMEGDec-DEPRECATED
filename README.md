# PyMEGDec

Utilities for MEG decoding experiments, including model transfer between experiment
conditions and cross-validation on a single dataset.
Reusable decoding summaries and prediction-table diagnostics are provided by
RepTrace; PyMEGDec keeps the MEG-specific loading, preprocessing, and workflow
entry points.

## Project boundary with RepTrace

PyMEGDec owns the dataset-specific MEG analysis layer. Keep MATLAB `.mat`
loading, `Part*Data.mat` / `Part*CueData.mat` participant-file conventions, CTF
sensor geometry handling, alpha analyses, stimulus-specific defaults, and
paper-facing scripts in this repository.

RepTrace owns the reusable M/EEG decoding layer. Feature-matrix decoding,
classifier calibration, temporal generalization, onset/state inference,
confusion and per-class metrics, MNE `Epochs` decoding, and generic
summary-table/reporting helpers should live in RepTrace and be imported here
rather than duplicated.

When adding new functionality, adapt PyMEGDec data into RepTrace's
feature-matrix or probability-observation interfaces instead of duplicating
reusable decoding logic here.

## Repository layout

```text
src/pymegdec/              Package source code
  alpha_signal.py          Alpha-band filtering and phase extraction
  alpha_metrics.py         Per-trial alpha power and phase-gradient export
  alpha_movement.py        Sensor-level alpha movement trajectory export
  alpha_visualization.py   Alpha signal and phase-shift plotting helpers
  reaction_time_analysis.py Alpha/RT join and association summaries
  stimulus_decoding.py     Time-resolved train-main / validate-cue decoding
  classifiers.py           Classifier factories and PyTorch Lightning model
  preprocessing.py         Filtering, downsampling, window extraction
  model_transfer.py        Train-on-experiment / validate-on-cue evaluation
  cross_validation.py      Single-dataset cross-validation routine
tests/                     Unit and data-dependent unittest suites
.github/workflows/         CI jobs for unit and data-dependent test subsets
```

Top-level `cross_validation.py`, `evaluate_model_transfer.py`,
`extract_alpha_signal.py`, `show_bandpass_signal_and_shifts.py`, and
`export_alpha_metrics.py` are compatibility wrappers for existing imports and
direct script usage. `analyze_alpha_reaction_time.py` provides an exploratory
analysis command for alpha metrics and behavioral reaction times.
`analyze_stimulus_decoding.py` exports time-resolved stimulus decoding curves.
`analyze_alpha_movement.py` exports sensor-level alpha movement trajectories.

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install poetry
poetry install
```

Install optional classifier backends when needed:

```bash
poetry install --extras "all"
```

## Data directory

Participant data is configured at runtime so private or machine-specific paths
do not need to be committed. Data files are expected to be named like
`Part2Data.mat` and `Part2CueData.mat`.

Resolution order:

1. Pass `--data-dir` to a CLI command, or pass `data_folder` to the Python API.
2. Set the `PYMEGDEC_DATA_DIR` environment variable.
3. Create a local `.pymegdec-data-dir` file containing one path. The resolver
   searches the current directory, its parents, and the project root.
4. Fall back to the current working directory for backwards compatibility.

`.pymegdec-data-dir` is ignored by git and can contain a path relative to the
file location.

On PowerShell:

```powershell
$env:PYMEGDEC_DATA_DIR = "C:\path\to\data"
python -m unittest
```

## CLI usage

```bash
pymegdec-cross-validate --data-dir "/path/to/MEG-Data" --participant 2
pymegdec-transfer --data-dir "/path/to/MEG-Data" --participant 2 --null-window-center nan
pymegdec-stimulus-decoding --data-dir "/path/to/MEG-Data" --participants 2 --output outputs/part2_stimulus_decoding.csv
```

The grouped command exposes the same workflows:

```bash
pymegdec cross-validate --participant 2
pymegdec transfer --participant 2 --classifier multiclass-svm
pymegdec stimulus-decoding --participants 2 --output outputs/part2_stimulus_decoding.csv
```

## Examples

```python
from pymegdec.model_transfer import evaluate_model_transfer
from pymegdec.cross_validation import cross_validate_single_dataset

transfer_accuracy = evaluate_model_transfer("/path/to/MEG-Data", 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset("/path/to/MEG-Data", 2, classifier="multiclass-svm")
```

If `PYMEGDEC_DATA_DIR` or `.pymegdec-data-dir` is configured, the first argument
can be `None`:

```python
transfer_accuracy = evaluate_model_transfer(None, 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset(None, 2, classifier="multiclass-svm")
```

## Time-resolved stimulus decoding

Stimulus decoding can be evaluated over a sequence of windows around stimulus
onset. The command trains on `Part*Data.mat`, validates on `Part*CueData.mat`,
and reports 16-way stimulus accuracy for each participant and window center.
Pass `--transfer-direction cue-to-main` to swap the direction and train on cue
data while validating on the main experiment data.
By default it uses no null class, because the cue validation files do not
contain null trials and the clean question is which of the 16 stimuli was shown.

```powershell
python analyze_stimulus_decoding.py --participants 2 --time-window=-0.2,0.6 --window-step-s 0.05 --output outputs\part2_stimulus_decoding.csv --summary-output outputs\part2_stimulus_decoding_summary.csv --plots-dir outputs\part2_stimulus_decoding_plots
```

The output CSV includes the stimulus-decoding accuracy, chance level, PCA
variance, and class counts for every participant/window row. Add
`--permutations N` to run label-shuffle tests (`N=0` by default, no permutation
step). Summary CSV adds `n_significant_p_0.05` and `n_significant_p_0.01` and
`n_with_permutation` per window.

```powershell
python analyze_stimulus_decoding.py `
  --participants 2 `
  --time-window=-0.2,0.6 `
  --window-step-s 0.05 `
  --permutations 200 `
  --permutation-seed 42 `
  --output outputs\part2_stimulus_decoding.csv `
  --summary-output outputs\part2_stimulus_decoding_summary.csv `
  --plots-dir outputs\part2_stimulus_decoding_plots
```

Peak-window diagnostics can be exported for selected windows. These write
trial-level predictions, confusion counts, per-stimulus accuracy, and
participant-level peak timing/accuracy.

```powershell
python analyze_stimulus_decoding.py `
  --participants 2 `
  --time-window=-0.2,0.6 `
  --window-step-s 0.05 `
  --diagnostic-window-centers 0.15,0.2,0.25 `
  --predictions-output outputs\part2_stimulus_predictions.csv `
  --confusion-output outputs\part2_stimulus_confusion.csv `
  --per-stimulus-output outputs\part2_stimulus_per_stimulus.csv `
  --participant-peaks-output outputs\part2_stimulus_participant_peaks.csv `
  --output outputs\part2_stimulus_decoding.csv `
  --summary-output outputs\part2_stimulus_decoding_summary.csv `
  --plots-dir outputs\part2_stimulus_decoding_plots
```

The summary CSV and plot aggregate the curve across participants.

For paper-facing confusion matrices, export only trial-level predictions at
selected control/peak windows. Model labels are zero-based when
`--null-window-center nan`, while `true_stimulus_id` and
`predicted_stimulus_id` are one-based image ids.

```powershell
python scripts\export_stimulus_predictions.py `
  --window-centers=-0.175,0.175 `
  --output outputs\stimulus_predictions.csv `
  --summary-output outputs\stimulus_prediction_summary.csv `
  --confusion-output outputs\stimulus_predictions_confusion.csv `
  --per-stimulus-output outputs\stimulus_predictions_per_stimulus.csv
```

For controls-first robustness checks, export the default two-window result plus
reverse transfer, class-balanced SVM, PCA sensitivity, and 0-30 Hz controls:

```powershell
python scripts\export_stimulus_robustness.py `
  --predictions-output outputs\stimulus_robustness_predictions.csv `
  --accuracy-output outputs\stimulus_robustness_accuracy.csv `
  --summary-output outputs\stimulus_robustness_summary.csv
```

Temporal generalization trains a separate model at each training window and
tests each model across all validation windows. This produces a train-time by
test-time matrix for assessing whether image information is transient or
generalizes across post-stimulus time.

```powershell
python scripts\export_stimulus_temporal_generalization.py `
  --participants 2 `
  --time-window=-0.4,0.8 `
  --window-step-s 0.025 `
  --output outputs\part2_stimulus_temporal_generalization.csv `
  --summary-output outputs\part2_stimulus_temporal_generalization_summary.csv
```

Onset-blind scanning trains one classifier at a known post-stimulus window and
slides it over validation trials. This is a pseudo-continuous test with the
current epoched data: the model is not given `t=0` while scanning, but the
known onset is still used afterward to score detection latency and false alarms.

```powershell
python scripts\export_stimulus_onset_scan.py `
  --participants 2 `
  --train-window-center 0.175 `
  --scan-time-window=-0.4,0.8 `
  --window-step-s 0.025 `
  --output outputs\part2_stimulus_onset_scan.csv `
  --events-output outputs\part2_stimulus_onset_events.csv `
  --summary-output outputs\part2_stimulus_onset_scan_summary.csv `
  --event-summary-output outputs\part2_stimulus_onset_event_summary.csv
```

## Tests

The default suite includes fast tests that run without private MEG files.
Data-dependent accuracy checks are skipped when the data directory cannot be
resolved.

```bash
python -m unittest discover -v
```

To run the data-dependent integration tests, point `PYMEGDEC_DATA_DIR` at a
directory containing `Part2Data.mat` and `Part2CueData.mat` before running the
same command.

## Exploratory alpha metrics

Prestimulus alpha metrics can be exported per trial for downstream plotting or
statistics. By default the exporter uses the `MLO*`, `MRO*`, and `MZO*`
occipital CTF channels and the `-0.4` to `-0.05 s` window before stimulus
onset.

```powershell
python export_alpha_metrics.py --participant 2 --output outputs\part2_alpha_metrics.csv
python export_alpha_metrics.py --participant 2 --cue --output outputs\part2_cue_alpha_metrics.csv
```

The exported rows include alpha power, phase concentration, planar phase-fit
quality, spatial phase frequency, estimated propagation speed, and dominant
phase-gradient direction on a projected sensor plane. The `outputs/` directory
is ignored by git.

## Sensor-level alpha movement

The MAT files contain CTF sensor geometry in `data.grad.chanpos`, with positions
in millimeters. This supports sensor-array analyses of alpha topography, but not
source-localized brain movement. The movement exporter therefore tracks a
sensor-level proxy: for each trial and sampled time point, it filters the chosen
MEG channels to the alpha band, computes alpha power, and writes the
power-weighted centroid over the MEG sensor positions.

By default it uses all MEG channels matching `^M`, the `8-12 Hz` alpha band, and
a `-0.4` to `0.8 s` window around stimulus onset.

```powershell
python analyze_alpha_movement.py --participants 2 --trajectory-output outputs\part2_alpha_movement.csv --summary-output outputs\part2_alpha_movement_summary.csv
```

The trajectory CSV includes 3D CTF sensor centroids, projected 2D centroids,
stepwise speed, displacement from the first sampled time point, the peak-power
channel, and a spatial concentration score. Treat the trajectory as movement of
the measured alpha topography over sensors, not as anatomical source motion.

Movement summaries can be analyzed into pre/post stimulus effects and simple
condition-level plots:

```powershell
python analyze_alpha_movement_results.py --movement-summary outputs\part2_alpha_movement_summary.csv --effect-output outputs\part2_alpha_movement_effects.csv --condition-summary-output outputs\part2_alpha_movement_condition_summary.csv --plots-dir outputs\part2_alpha_movement_plots
```

The effects compare the mean pre-stimulus centroid with the mean post-stimulus
centroid, and summarize speed, alpha power, and spatial concentration changes.

## Alpha and reaction time

The saved MEG `Part*Data.mat` files may not contain reaction times. The RT
analysis command therefore accepts an external behavioral CSV with
`participant`, `trial`, and `reaction_time` columns. If reaction times are stored
in a future MAT `trialinfo` column, pass `--trialinfo-rt-column` instead.

```powershell
python analyze_alpha_reaction_time.py --participants 2 --reaction-times behavior_rt.csv --joined-output outputs\part2_alpha_rt_trials.csv --summary-output outputs\part2_alpha_rt_summary.csv
```

The summary includes per-participant Pearson/regression rows and a pooled
within-participant row for each alpha metric. Phase-gradient direction is encoded
as sine and cosine components before analysis.
