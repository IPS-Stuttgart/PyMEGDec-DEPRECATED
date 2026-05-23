# Covariance-feature stimulus decoding

This source-only LOSO benchmark tests whether trial covariance structure carries
cross-subject visual-stimulus information that is missed by the historical
sensor-flat amplitude features.

The command uses only `Part*Data.mat` main-task files. It does not load
`Part*CueData.mat`, so cue/localizer files remain reserved for calibration-only
experiments.

## Recommended first run

```bash
pymegdec stimulus cross-subject-covariance \
  --participants 1-4,6,8,9,10,13-27 \
  --time-windows 0.05:0.30,0.08:0.35,0.05:0.45 \
  --baseline-window -0.35:-0.05 \
  --normalizations subject_baseline_whiten,subject_baseline_z \
  --feature-modes logeuclidean_covariance,correlation_upper,variance \
  --covariance-shrinkages 0.05,0.1,0.3 \
  --projections pca \
  --components-values 32,64,128 \
  --classifiers multinomial-logistic,multinomial-logistic-weighted \
  --classifier-params 0.03,0.1,0.3,1,3
```

The strict evaluation path is:

```text
main-task Part*Data.mat only
→ per-subject baseline z/whitening on channel time courses
→ trial covariance inside each candidate post-stimulus window
→ log-Euclidean / covariance / correlation / variance vectorization
→ optional PCA or PLS fitted on source subjects only
→ linear multiclass classifier
→ nested source-subject LOSO model selection
→ untouched held-out-subject evaluation
```

## Feature modes

- `logeuclidean_covariance`: shrink the trial covariance to SPD, apply a matrix
  logarithm, then vectorize the upper triangle with off-diagonal `sqrt(2)`
  scaling. This is the closest dependency-free Riemannian-style feature.
- `covariance_upper`: vectorized shrunk covariance upper triangle.
- `correlation_upper`: vectorized covariance converted to correlation.
- `variance`: log diagonal variance only; this is a cheap sanity check against
  the full covariance modes.

`--covariance-shrinkages` controls shrinkage toward scaled identity before
vectorization. `--covariance-epsilons` controls the positive eigenvalue floor,
relative to mean variance, used for stable matrix logarithms.

## Null control

```bash
pymegdec stimulus cross-subject-covariance \
  --participants 1-4,6,8,9,10,13-27 \
  --time-windows 0.05:0.30 \
  --baseline-window -0.35:-0.05 \
  --normalizations subject_baseline_whiten \
  --feature-modes logeuclidean_covariance \
  --covariance-shrinkages 0.1 \
  --projections pca \
  --components-values 64,128 \
  --classifiers multinomial-logistic \
  --classifier-params 0.1,1,3 \
  --label-shuffle-control \
  --outer-output outputs/covariance_label_shuffle_outer.csv \
  --summary-output outputs/covariance_label_shuffle_group_summary.csv
```

The shuffled-label run should return group performance near 16-way chance.
