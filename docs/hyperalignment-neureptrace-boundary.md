# NeuRepTrace boundary for hyperalignment

PyMEGDec keeps the BUSH-MEG-specific cross-subject hyperalignment workflow: MAT-file loading, feature extraction, LOSO fold orchestration, output naming, and compatibility CLI arguments.

The reusable alignment kernels are bound to NeuRepTrace:

- `neureptrace.decoding.hyperalignment.class_alignment_matrices`
- `neureptrace.decoding.hyperalignment.fit_class_hyperalignment`
- `neureptrace.decoding.hyperalignment.fit_projection_to_hyperalignment`
- `neureptrace.decoding.hyperalignment_initialization`
- `neureptrace.decoding.windowed` for the fitted shared-space decoder helpers

This means new generic improvements to Procrustes hyperalignment should be made in NeuRepTrace. PyMEGDec should only adapt BUSH-MEG data and preserve old command names such as:

```bash
pymegdec stimulus cross-subject-hyperalignment \
  --participants 1-4,6,8,9,10,13-27 \
  --window-center 0.175 \
  --window-size 0.1 \
  --feature-mode sensor_flat \
  --normalization subject_baseline_z \
  --classifier multiclass-svm \
  --components-pca 64 \
  --hyper-components 64
```
