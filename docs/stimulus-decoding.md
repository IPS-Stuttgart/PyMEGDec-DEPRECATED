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
against 16-way chance. The group summary also reports an exact one-sided sign
test over held-out participants, so the direction-only result is visible next
to the magnitude-sensitive sign-flip p-value.

The manual GitHub Actions smoke workflow has a `benchmark_mode` input. Use
`poststimulus` for the standard `0.175` s benchmark and
`prestimulus-control` for an explicit `-0.175` s control. The workflow writes
self-describing artifacts such as
`stimulus_cross_subject_prestimulus_control_outer.csv` and
`stimulus_cross_subject_prestimulus_control_group_summary.csv`.

Useful follow-up classifiers for this cross-subject benchmark are
`correlation-prototype`, `multinomial-logistic`, and `shrinkage-lda`.
`correlation-prototype` classifies each held-out trial by correlation to the
training-set class-average pattern, which is a simple baseline for shared
stimulus topographies across participants.
For `multinomial-logistic`, `--classifier-params` controls the inverse
regularization strength `C`. For `shrinkage-lda`, use `auto` or a numeric
shrinkage value between `0` and `1`.

Supported cross-subject normalization modes are `none`, `subject_z`,
`subject_trial_z`, `subject_baseline_z`, and `subject_baseline_whiten`.
`subject_trial_z` normalizes each trial feature vector independently.
`subject_baseline_whiten` estimates a shrinkage channel covariance from the
baseline window and applies the resulting channel whitening transform to each
stimulus-window feature vector.

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
  --normalizations subject_baseline_z,subject_trial_z,subject_baseline_whiten \
  --alignments none,train_class_procrustes \
  --classifiers multinomial-logistic,shrinkage-lda,multiclass-svm \
  --classifier-params default \
  --components-pca-values 64 \
  --max-trials-per-class-per-participant 10
```

The trial cap is a deterministic screening option: by default it draws a seeded
random subset of `N` trials from each stimulus class for each participant,
preserving nested LOSO while avoiding a file-order or block-order bias in fast
candidate-selection runs. Use `--trial-selection-seed` to reproduce a screening
subset, and use `--trial-selection first` only when you intentionally need the
legacy first-trials-per-class behavior. Omit `--max-trials-per-class-per-participant`
for the final all-trial benchmark.
In the manual GitHub Actions workflow, set `benchmark_preset` to
`final-all-trials` for the final run. That preset ignores the trial-cap input,
does not pass `--max-trials-per-class-per-participant`, and writes outputs with
the `stimulus_cross_subject_nested_final_all_trials_*` prefix so they cannot be
confused with screening artifacts. Keep `benchmark_preset=screening` for fast
grid checks that intentionally use the trial cap.

The nested outputs include untouched outer-fold scores, one row per inner
validation fold and candidate, selected hyperparameters per outer fold, trial
predictions, confusion counts, per-stimulus accuracy, and a group summary.
The group summary reports selected-candidate counts for classifier, window
center, feature mode, normalization, alignment, and PCA components. It also
reports the inner winner margin, defined per outer fold as the best inner
balanced accuracy minus the second-best inner balanced accuracy, to make noisy
hyperparameter selection visible.

The `train_class_procrustes` alignment mode fits an orthogonal channel-space
Procrustes transform from the outer-training participants only. It uses
training-participant class-average topographies to align the training pool to a
shared template, then evaluates the held-out participant without using that
participant's labels for alignment. This is a conservative first alignment
control; supervised held-out-subject alignment would require a separate labeled
calibration design.

## Cue-calibrated cross-subject benchmark

The cue-calibrated workflow tests that separate calibration design. It leaves
one participant out for the scored main-task decoding, but uses each
participant's independent `Part*CueData.mat` file to estimate an orthogonal
channel-space Procrustes transform into a source-only cue template. The source
template is built from the outer-training participants only; the held-out
participant's cue data are used only to fit that participant's transform. The
main-task decoder is still trained on source participants' `Part*Data.mat`
labels and scored on the held-out participant's `Part*Data.mat` trials.

```bash
pymegdec stimulus cross-subject-cue-calibrated \
  --participants 1-4,6,8,9,10,13-27 \
  --window-center 0.175 \
  --window-size 0.1 \
  --feature-mode sensor_flat \
  --normalization subject_baseline_z \
  --classifier multiclass-svm \
  --components-pca 64 \
  --calibration-feature-mode decode \
  --calibration-normalization decode
```

By default, the cue calibration window, window size, baseline window, feature
mode, and normalization reuse the main decoding settings. Override
`--calibration-window-center`, `--calibration-window-size`,
`--calibration-baseline-window`, `--calibration-feature-mode`, or
`--calibration-normalization` when the localizer has a different timing or when
you want a deliberately different calibration representation.

Two leakage controls are available. `--label-shuffle-control` shuffles the
source participants' main-task training labels while keeping scoring labels
untouched. `--target-calibration-label-shuffle-control` shuffles the held-out
participant's cue labels before fitting that participant's transform, testing
whether the target-side class correspondence is carrying the improvement.

Add `--label-shuffle-control` to run a nested null control with the same
participant splits and candidate grid, but with stimulus labels shuffled within
each training participant before every model fit. Validation and outer-test
labels remain untouched, so this control should return group performance near
16-way chance if the pipeline is not exploiting leakage or a nonstimulus
confound. In the manual workflow, enable `label_shuffle_control`; artifacts are
written as `stimulus_cross_subject_nested_label_shuffle_*.csv`.

## Cross-subject shared-space alignment benchmarks

Use the manual `stimulus-cross-subject-alignment-benchmarks.yml` workflow to
compare RepTrace-backed Procrustes hyperalignment and M-CCA on the same
`Part*Data.mat` LOSO image-identity task. This workflow is separate from the
nested candidate-grid workflow because hyperalignment and M-CCA use dedicated
shared-space fitting code and expose additional method-specific parameters.

The grouped commands are also available locally:

```bash
pymegdec stimulus cross-subject-hyperalignment \
  --participants 1-4,6,8,9,10,13-27 \
  --window-center 0.175 \
  --window-size 0.1 \
  --feature-mode sensor_flat \
  --normalization subject_baseline_z \
  --classifier multiclass-svm \
  --components-pca 64 \
  --hyper-components 64 \
  --target-centering target_unsupervised
```

```bash
pymegdec stimulus cross-subject-mcca \
  --participants 1-4,6,8,9,10,13-27 \
  --window-center 0.175 \
  --window-size 0.1 \
  --feature-mode sensor_flat \
  --normalization subject_baseline_z \
  --classifier multiclass-svm \
  --components-pca 64 \
  --mcca-components 64 \
  --mcca-regularization 1e-6 \
  --target-centering target_unsupervised
```

The workflow default runs both `hyperalignment` and `mcca`, uses the same main
MEG data preparation action as the nested workflow, writes method-specific
outer-fold and group-summary CSVs, and uploads them as
`stimulus-cross-subject-alignment-outputs`. M-CCA additionally writes trial
predictions, confusion counts, per-stimulus recall, and confusion-pair summaries.

Both methods fit the shared space from the outer-training participants. The
default `target_unsupervised` option centers the held-out participant using its
own unlabeled feature mean before projecting into the group space; use
`group_mean` for a stricter target-independent centering control. Enable
`label_shuffle_control` to check whether the shared-space pipeline returns
near-chance performance when training labels are shuffled within participants.

Set `--alignment-data cue` to fit the M-CCA or hyperalignment projections from
the independent `Part*CueData.mat` files instead of the scored main-task files.
In that mode, source and held-out participants are projected using cue-file
class anchors, while the classifier is trained on source participants'
`Part*Data.mat` trials and evaluated on all held-out `Part*Data.mat` trials:

```bash
pymegdec stimulus cross-subject-mcca \
  --participants 1-4,6,8,9,10,13-27 \
  --window-center 0.175 \
  --window-size 0.1 \
  --alignment-data cue \
  --feature-mode sensor_flat \
  --normalization subject_baseline_z \
  --classifier multiclass-svm \
  --components-pca 64 \
  --mcca-components 64 \
  --mcca-regularization 1e-6
```

The manual alignment benchmark and M-CCA/hyperalignment grid workflows expose
the same `alignment_data` input. When `alignment_data=cue`, the workflows request
both `Part*Data.mat` and `Part*CueData.mat` from the configured data source and
require all 23 main/cue subject files before running.

## Confusion structure

Use the confusion-structure export on trial prediction CSVs to check whether
off-diagonal errors are meaningful rather than uniformly random. With stimulus
metadata, the export tests whether confused image pairs share categories more
often than expected from the true-category and predicted-category error
marginals.

```bash
python scripts/export_stimulus_confusion_structure.py \
  --predictions outputs/stimulus_cross_subject_nested_predictions.csv \
  --stimulus-metadata stimulus_metadata.csv \
  --output outputs/stimulus_cross_subject_nested_confusion_pairs.csv \
  --category-output outputs/stimulus_cross_subject_nested_category_enrichment.csv \
  --category-matrix-output outputs/stimulus_cross_subject_nested_category_matrix.csv \
  --category-columns visual_category,semantic_category
```

The metadata CSV should contain one stimulus id column such as `stimulus` or
`image_id`, plus one or more repeated category columns. The category-enrichment
CSV reports same-category error rate, expected rate, lift, standardized
residual, participant support, and a permutation p-value. The category matrix
CSV reports directional category-to-category error counts and lifts.

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
