# Covariance-feature stimulus decoding

PyMEGDec no longer owns the covariance-feature implementation. The historical
`pymegdec stimulus cross-subject-covariance` command is now a compatibility
wrapper around NeuRepTrace's `bushmeg-covariance-loso` workflow.

Use NeuRepTrace directly for new runs whenever possible:

```bash
neureptrace bushmeg-covariance-loso configs/bush_meg/covariance_loso.yml
```

The PyMEGDec command remains available for old scripts. It translates the most
common legacy command-line arguments into a temporary NeuRepTrace JSON config,
then calls `neureptrace.bushmeg_covariance_loso.run_bushmeg_covariance_loso`.

## Recommended direct NeuRepTrace workflow

The canonical covariance config lives in NeuRepTrace and describes a strict
source-only BUSH-MEG run:

```bash
neureptrace bushmeg-covariance-loso configs/bush_meg/covariance_loso.yml \
  --out results/bush_meg/covariance_loso/covariance_loso_summary.csv \
  --inner-cv-out results/bush_meg/covariance_loso/covariance_loso_inner_cv.csv \
  --predictions-out results/bush_meg/covariance_loso/covariance_loso_predictions.csv
```

That workflow loads only `Part*Data.mat` main-task files. It does not load
`Part*CueData.mat`, so cue/localizer files remain reserved for explicit
calibration workflows.

## Compatibility command

The old PyMEGDec command still works for straightforward runs:

```bash
pymegdec stimulus cross-subject-covariance \
  --participants 1-4,6,8,9,10,13-27 \
  --time-windows 0.05:0.30,0.08:0.35,0.05:0.45 \
  --baseline-window -0.35:-0.05 \
  --normalizations subject_baseline_whiten \
  --feature-modes logeuclidean_covariance,correlation_upper,variance \
  --covariance-shrinkages 0.05,0.1,0.3 \
  --projections pca \
  --components-values 32,64,128 \
  --classifiers multinomial-logistic \
  --classifier-params 0.03,0.1,0.3,1,3
```

To inspect the generated NeuRepTrace config, add:

```bash
pymegdec stimulus cross-subject-covariance \
  --write-neureptrace-config outputs/generated_covariance_loso.json
```

To run a hand-written NeuRepTrace config through the old PyMEGDec entry point:

```bash
pymegdec stimulus cross-subject-covariance \
  --neureptrace-config configs/bush_meg/covariance_loso.yml
```

## Migration notes

The compatibility wrapper intentionally keeps only lightweight translation and
CSV-compatibility helpers in PyMEGDec. The following are now NeuRepTrace-owned:

- covariance feature vectors;
- covariance feature-mode normalization;
- covariance candidate specifications;
- FieldTrip/MAT loading for this workflow;
- nested source-only LOSO model selection;
- held-out-subject scoring and probability output.

Some legacy PyMEGDec options cannot be translated one-to-one, such as evaluating
only selected outer participants or deterministic trial caps. For those cases,
write or modify a native NeuRepTrace `covariance_loso` config instead of adding
new logic to PyMEGDec.

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
