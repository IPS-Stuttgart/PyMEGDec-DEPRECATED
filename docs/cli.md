# CLI reference

PyMEGDec exposes one grouped command. Prefer the grouped `pymegdec` command for
new documentation, CI jobs, and shell scripts. Compatibility aliases and
historical Python wrappers remain available for existing workflows.

```bash
pymegdec --help
```

## Command groups

### Core decoding

```bash
pymegdec cross-validate --participant 2
pymegdec transfer --participant 2 --classifier multiclass-svm
pymegdec stimulus-decoding --participants 2 --output outputs/part2_stimulus_decoding.csv
pymegdec make-synthetic-data --out demo-data
pymegdec alpha-movement-results --movement-summary outputs/part2_alpha_movement_summary.csv --effect-output outputs/part2_alpha_movement_effects.csv --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv
```

### Stimulus workflows

```bash
pymegdec stimulus decoding --participants 2 --output outputs/part2_stimulus_decoding.csv
pymegdec stimulus cross-subject-smoke --participants 1-4,6,8,9,10,13-27 --outer-output outputs/stimulus_cross_subject_outer.csv
pymegdec stimulus cross-subject-cue-calibrated --participants 1-4,6,8,9,10,13-27 --feature-mode sensor_flat --normalization subject_baseline_z
pymegdec stimulus cross-subject-nested --participants 1-4,6,8,9,10,13-27 --window-centers 0.150,0.175,0.200 --classifiers multinomial-logistic,shrinkage-lda,multiclass-svm --max-trials-per-class-per-participant 10
pymegdec stimulus predictions --participants 2 --output outputs/stimulus_predictions.csv
pymegdec stimulus robustness --participants 2 --predictions-output outputs/stimulus_robustness_predictions.csv
pymegdec stimulus temporal-generalization --participants 2 --output outputs/stimulus_temporal_generalization.csv
pymegdec stimulus onset-scan --participants 2 --output outputs/stimulus_onset_scan.csv --events-output outputs/stimulus_onset_events.csv
```

### Alpha workflows

```bash
pymegdec alpha metrics --participant 2 --output outputs/part2_alpha_metrics.csv
pymegdec alpha movement --participants 2 --trajectory-output outputs/part2_alpha_movement.csv --summary-output outputs/part2_alpha_movement_summary.csv
pymegdec alpha movement-results --movement-summary outputs/part2_alpha_movement_summary.csv --effect-output outputs/part2_alpha_movement_effects.csv --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv
pymegdec alpha reaction-time --participants 2 --joined-output outputs/part2_alpha_rt_joined.csv --summary-output outputs/part2_alpha_rt_summary.csv
```

### Data helpers

```bash
pymegdec data download --data-dir data --env-name MEG_DATA_URL_LIST
pymegdec data download --source webdav-rclone --data-dir data --file-indices 2,3
pymegdec data download --source webdav-rclone --data-dir data --file-names Part2CueData.mat,Part2Data.mat
```

## Backward-compatible aliases

The following top-level aliases are kept so old commands and existing CI jobs do
not need to change immediately:

```bash
pymegdec stimulus-decoding
pymegdec stimulus-cross-subject-cue-calibrated
pymegdec stimulus-cross-subject-nested
pymegdec stimulus-cross-subject-smoke
pymegdec stimulus-predictions
pymegdec stimulus-robustness
pymegdec stimulus-temporal-generalization
pymegdec stimulus-onset-scan
pymegdec alpha-metrics
pymegdec alpha-movement
pymegdec alpha-movement-results
pymegdec alpha-reaction-time
pymegdec alpha-rt
pymegdec download-meg-data
```

The installed script names remain available:

```bash
pymegdec-cross-validate
pymegdec-transfer
pymegdec-stimulus-decoding
pymegdec-stimulus-cross-subject-cue-calibrated
pymegdec-alpha-movement-results
pymegdec-make-synthetic-data
```

Top-level Python wrappers such as `python export_alpha_metrics.py` and
`scripts/export_stimulus_predictions.py` now delegate to the package-level
command handlers.

## Shared decoding options

The cross-validation and transfer commands share the core decoding options:

| Option                       | Meaning                                     | Typical value       |
|------------------------------|---------------------------------------------|---------------------|
| `--data-dir`                 | Directory containing participant MAT files. | `/path/to/MEG-Data` |
| `--participant`              | Participant id.                             | `2`                 |
| `--window-size`              | Window duration in seconds.                 | `0.1`               |
| `--train-window-center`      | Stimulus training-window center in seconds. | `0.2`               |
| `--null-window-center`       | Null-window center, or `nan`.               | `nan` or `-0.2`     |
| `--new-framerate`            | Target frame rate, or `inf`.                | `inf`               |
| `--classifier`               | Classifier registry name.                   | `multiclass-svm`    |
| `--classifier-param`         | Numeric, JSON, or Python literal parameter. | `1.0`               |
| `--components-pca`           | PCA component count, or `inf`.              | `100`               |
| `--frequency-range LOW HIGH` | Frequency range in Hz.                      | `0 inf`             |

## Synthetic demo data

Generate balanced private-data-free MATLAB files for participant 2:

```bash
pymegdec make-synthetic-data --out demo-data
```

The generator writes `Part2Data.mat`, `Part2CueData.mat`, and
`synthetic_data_manifest.json`. The generated files follow the same participant
file naming convention as the private MEG data and can be used with the standard
commands:

```bash
pymegdec cross-validate --data-dir demo-data --participant 2
pymegdec transfer --data-dir demo-data --participant 2 --null-window-center nan
```

Customize the generated dataset when testing edge cases:

```bash
pymegdec make-synthetic-data \
  --out demo-data \
  --participant 3 \
  --classes 4 \
  --main-repeats 8 \
  --cue-repeats 4 \
  --channels 6 \
  --overwrite
```

## Cross-validation

Cross-validate one participant's main dataset:

```bash
pymegdec cross-validate --data-dir /path/to/MEG-Data --participant 2 --folds 10
```

## Model transfer

Train on the main experiment file and validate on the cue file for one
participant:

```bash
pymegdec transfer --data-dir /path/to/MEG-Data --participant 2 --null-window-center nan
```

## Stimulus decoding

Run train-main / validate-cue decoding across a time range:

```bash
pymegdec stimulus decoding \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --time-window=-0.2,0.6 \
  --window-step-s 0.05 \
  --output outputs/part2_stimulus_decoding.csv \
  --summary-output outputs/part2_stimulus_decoding_summary.csv \
  --plots-dir outputs/part2_stimulus_decoding_plots
```

Use `--transfer-direction cue-to-main` to train on cue data and validate on the
main experiment data.

## Stimulus prediction diagnostics

Export trial-level predictions and optional confusion/per-stimulus summaries for
selected diagnostic windows:

```bash
pymegdec stimulus predictions \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --window-centers=-0.175,0.175 \
  --output outputs/stimulus_predictions.csv \
  --summary-output outputs/stimulus_prediction_summary.csv \
  --confusion-output outputs/stimulus_confusion.csv \
  --per-stimulus-output outputs/stimulus_per_class.csv
```

## Stimulus robustness controls

```bash
pymegdec stimulus robustness \
  --data-dir /path/to/MEG-Data \
  --participants 1-4,6,8,9,10,13-27 \
  --window-centers=-0.175,0.175 \
  --predictions-output outputs/stimulus_robustness_predictions.csv \
  --accuracy-output outputs/stimulus_robustness_accuracy.csv \
  --summary-output outputs/stimulus_robustness_summary.csv
```

## Stimulus temporal generalization

```bash
pymegdec stimulus temporal-generalization \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --time-window=-0.4,0.8 \
  --window-step-s 0.025 \
  --output outputs/stimulus_temporal_generalization.csv \
  --summary-output outputs/stimulus_temporal_generalization_summary.csv
```

## Stimulus onset scan

```bash
pymegdec stimulus onset-scan \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --scan-time-window=-0.4,0.8 \
  --threshold-window=-0.35,-0.05 \
  --output outputs/stimulus_onset_scan.csv \
  --events-output outputs/stimulus_onset_events.csv
```

## Alpha movement result analysis

Analyze a movement summary exported by `pymegdec alpha movement`:

```bash
pymegdec alpha movement-results \
  --movement-summary outputs/part2_alpha_movement_summary.csv \
  --effect-output outputs/part2_alpha_movement_effects.csv \
  --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv \
  --plots-dir outputs/part2_alpha_movement_plots
```
