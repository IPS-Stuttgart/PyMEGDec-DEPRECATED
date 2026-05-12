# Stimulus decoding workflows

Stimulus decoding evaluates image identity over time. The default direction
trains on `Part*Data.mat`, validates on `Part*CueData.mat`, and reports 16-way
stimulus accuracy for each participant and window center.

Use `--transfer-direction cue-to-main` to swap the direction and train on cue
data while validating on the main experiment data.

By default, stimulus decoding uses no null class because the cue validation
files do not contain null trials; the direct question is which of the 16 stimuli
was shown.

## Cross-subject smoke test

The cross-subject smoke workflow leaves one participant out, trains on all
other participants, and tests 16-way image identity on the held-out participant.
It uses `Part*Data.mat` files only; `Part*CueData.mat` files are not loaded.

```bash
pymegdec stimulus cross-subject-smoke \
  --participants 1-4,6,8,9,10,13-27 \
  --window-center 0.175 \
  --window-size 0.1 \
  --normalization subject_baseline_z \
  --components-pca 64
```

The first-pass feature mode averages every sensor within the decoding window,
then fits a linear multiclass SVM after training-fold PCA. The outputs include
held-out participant scores, trial predictions, confusion counts, per-stimulus
recall, and a group summary with a subject-level one-sided sign-flip test
against 16-way chance.

Useful follow-up classifiers for this cross-subject benchmark are
`correlation-prototype`, `multinomial-logistic`, and `shrinkage-lda`.
`correlation-prototype` classifies each held-out trial by correlation to the
training-set class-average pattern, which is a simple baseline for shared
stimulus topographies across participants.

## Nested cross-subject benchmark

Use nested LOSO when choosing among windows, classifiers, or PCA settings. For
each outer participant, PyMEGDec leaves that participant untouched, selects the
best candidate by inner leave-one-subject-out validation on the remaining
participants, then refits the selected candidate on all outer-training
participants before scoring the held-out participant.

```bash
pymegdec stimulus cross-subject-nested \
  --participants 1-4,6,8,9,10,13-27 \
  --window-centers 0.150,0.175,0.200 \
  --window-size 0.1 \
  --feature-modes sensor_flat \
  --normalizations subject_baseline_z \
  --classifiers multinomial-logistic,shrinkage-lda,multiclass-svm \
  --classifier-params default \
  --components-pca-values 64
```

The nested outputs include untouched outer-fold scores, one row per inner
validation fold and candidate, selected hyperparameters per outer fold, trial
predictions, confusion counts, per-stimulus accuracy, and a group summary.

## Time-resolved decoding curve

```powershell
python analyze_stimulus_decoding.py `
  --participants 2 `
  --time-window=-0.2,0.6 `
  --window-step-s 0.05 `
  --output outputs\part2_stimulus_decoding.csv `
  --summary-output outputs\part2_stimulus_decoding_summary.csv `
  --plots-dir outputs\part2_stimulus_decoding_plots
```

The participant/window CSV includes decoding accuracy, chance level, PCA
variance, and class counts. The summary CSV aggregates across participants by
window center.

Equivalent grouped command:

```bash
pymegdec stimulus-decoding \
  --participants 2 \
  --time-window=-0.2,0.6 \
  --window-step-s 0.05 \
  --output outputs/part2_stimulus_decoding.csv \
  --summary-output outputs/part2_stimulus_decoding_summary.csv \
  --plots-dir outputs/part2_stimulus_decoding_plots
```

## Permutation p-values

Add label-shuffle tests with `--permutations N`. `N=0` disables the permutation
step.

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

The summary output adds `n_significant_p_0.05`, `n_significant_p_0.01`, and
`n_with_permutation` per window.

## Peak-window diagnostics

Export trial-level predictions, confusion counts, per-stimulus accuracy, and
participant-level peak timing for selected windows:

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

When diagnostics are requested through `pymegdec stimulus-decoding`, pass
`--diagnostic-window-centers` together with the diagnostic output paths.

## Paper-facing prediction exports

For confusion matrices and paper-facing trial-level tables, export predictions
at selected control and peak windows:

```powershell
python scripts\export_stimulus_predictions.py `
  --window-centers=-0.175,0.175 `
  --output outputs\stimulus_predictions.csv `
  --summary-output outputs\stimulus_prediction_summary.csv `
  --confusion-output outputs\stimulus_predictions_confusion.csv `
  --per-stimulus-output outputs\stimulus_predictions_per_stimulus.csv
```

Model labels are zero-based when `--null-window-center nan`. The exported
`true_stimulus_id` and `predicted_stimulus_id` columns use one-based image ids.

## Robustness checks

The controls-first robustness export writes the default two-window result plus
reverse transfer, class-balanced SVM, PCA sensitivity, and 0–30 Hz controls:

```powershell
python scripts\export_stimulus_robustness.py `
  --predictions-output outputs\stimulus_robustness_predictions.csv `
  --accuracy-output outputs\stimulus_robustness_accuracy.csv `
  --summary-output outputs\stimulus_robustness_summary.csv
```

## Temporal generalization

Temporal generalization trains a separate model at each training window and
tests each model across all validation windows. The output is a train-time by
test-time matrix for assessing whether stimulus information is transient or
generalizes across post-stimulus time.

```powershell
python scripts\export_stimulus_temporal_generalization.py `
  --participants 2 `
  --time-window=-0.4,0.8 `
  --window-step-s 0.025 `
  --output outputs\part2_stimulus_temporal_generalization.csv `
  --summary-output outputs\part2_stimulus_temporal_generalization_summary.csv
```

## Onset-blind scanning

Onset scanning trains one classifier at a known post-stimulus window and slides
it over validation trials. With the current epoched data this is a
pseudo-continuous test: the model is not given `t=0` while scanning, but the
known onset is used afterward to score detection latency and false alarms.

```powershell
python scripts\export_stimulus_onset_scan.py `
  --participants 2 `
  --train-window-center 0.175 `
  --scan-time-window=-0.4,0.8 `
  --window-step-s 0.025 `
  --threshold-method max_run `
  --min-consecutive 2 `
  --min-duration 0.05 `
  --require-stable-prediction `
  --detection-start-s 0.0 `
  --output outputs\part2_stimulus_onset_scan.csv `
  --events-output outputs\part2_stimulus_onset_events.csv `
  --summary-output outputs\part2_stimulus_onset_scan_summary.csv `
  --event-summary-output outputs\part2_stimulus_onset_event_summary.csv
```

`--threshold-method max_run` estimates thresholds from sequence-level baseline
maxima under the same run criterion used by the detector. Combine it with
`--min-consecutive`, `--min-duration`, and `--require-stable-prediction` to
suppress one-bin spikes and class-flip artifacts while scanning many candidate
time bins.
