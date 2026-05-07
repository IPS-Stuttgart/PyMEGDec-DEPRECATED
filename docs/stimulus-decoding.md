# Stimulus decoding workflows

Stimulus decoding evaluates image identity over time. The default direction
trains on `Part*Data.mat`, validates on `Part*CueData.mat`, and reports 16-way
stimulus accuracy for each participant and window center.

Use `--transfer-direction cue-to-main` to swap the direction and train on cue
data while validating on the main experiment data.

By default, stimulus decoding uses no null class because the cue validation
files do not contain null trials; the direct question is which of the 16 stimuli
was shown.

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
  --output outputs\part2_stimulus_onset_scan.csv `
  --events-output outputs\part2_stimulus_onset_events.csv `
  --summary-output outputs\part2_stimulus_onset_scan_summary.csv `
  --event-summary-output outputs\part2_stimulus_onset_event_summary.csv
```
