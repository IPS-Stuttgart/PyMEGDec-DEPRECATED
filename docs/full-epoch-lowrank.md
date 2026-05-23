# Full-epoch low-rank stimulus decoding

The full-epoch low-rank benchmark is a source-only LOSO stimulus-decoding path
for testing whether a wider temporal representation improves over the historical
single-window PCA baseline.

The command uses only `Part*Data.mat` main-task files. It does not load
`Part*CueData.mat` and therefore keeps cue/localizer data reserved for separate
calibration experiments.

## Recommended first run

```bash
pymegdec stimulus cross-subject-full-epoch-lowrank \
  --participants 1-4,6,8,9,10,13-27 \
  --time-windows 0.00:0.45 \
  --time-bin-size 0.01 \
  --baseline-window -0.35:-0.05 \
  --temporal-feature-modes mean,mean+d1 \
  --normalizations subject_baseline_whiten \
  --projections pls \
  --components-values 32,64,128 \
  --classifiers multinomial-logistic \
  --classifier-params 0.03,0.1,0.3,1,3
```

This implements the practical baseline:

```text
subject baseline whitening
→ full-epoch crop
→ 10 ms channel × time bins
→ optional bin-to-bin temporal-difference block (`mean+d1`)
→ supervised PLS low-rank projection
→ multinomial logistic regression
→ nested source-subject LOSO selection
→ untouched held-out-subject evaluation
```

## Candidate grids

The command supports comma-separated candidate grids for:

- `--time-windows`, for example `0.05:0.35,0.00:0.45,-0.05:0.60,0.00:1.00`.
- `--temporal-feature-modes`, for example `mean,mean+d1`.
- `--normalizations`, for example `subject_baseline_whiten,subject_baseline_z`.
- `--projections`, currently `pls`, `pca`, or `none`.
- `--components-values`, for example `16,32,64,96,128`.
- `--classifiers`, for example `multinomial-logistic,multinomial-logistic-weighted,shrinkage-lda`.
- `--classifier-params`, for example `0.03,0.1,0.3,1,3`.

Each outer held-out participant is untouched during model selection. PyMEGDec
runs inner leave-one-subject-out validation over the remaining source subjects,
selects the best candidate by mean balanced accuracy, refits the selected
candidate on all source subjects, and scores the held-out subject once.

## Outputs

The workflow writes the same artifact family as the nested windowed benchmark:

- outer-fold scores;
- group summary;
- inner-validation scores;
- selected hyperparameters;
- trial predictions;
- confusion counts;
- per-stimulus recall;
- bidirectional confusion-pair summaries.

Rows include `time_window_s`, `time_bin_size_s`, `projection`, `n_components`,
`projection_actual_components`, `temporal_feature_mode`, top-2/top-3 accuracy,
and true-label ranks.

## Null control

Use the label-shuffle control to check that the full-epoch path does not exploit
metadata leakage or file-order artifacts:

```bash
pymegdec stimulus cross-subject-full-epoch-lowrank \
  --participants 1-4,6,8,9,10,13-27 \
  --time-windows 0.00:0.45 \
  --time-bin-size 0.01 \
  --baseline-window -0.35:-0.05 \
  --temporal-feature-modes mean,mean+d1 \
  --normalizations subject_baseline_whiten \
  --projections pls \
  --components-values 32,64,128 \
  --classifiers multinomial-logistic \
  --classifier-params 0.03,0.1,0.3,1,3 \
  --label-shuffle-control \
  --outer-output outputs/full_epoch_lowrank_label_shuffle_outer.csv \
  --summary-output outputs/full_epoch_lowrank_label_shuffle_group_summary.csv
```

The shuffled-label run should return group performance near 16-way chance.
