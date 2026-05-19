# Python API

Prefer importing from `pymegdec.*` modules instead of top-level compatibility
wrappers. The top-level scripts remain available for historical command-line and
notebook usage.

## Model transfer and cross-validation

```python
from pymegdec.model_transfer import evaluate_model_transfer
from pymegdec.cross_validation import cross_validate_single_dataset

transfer_accuracy = evaluate_model_transfer(
    "/path/to/MEG-Data",
    2,
    classifier="multiclass-svm",
)

cv_accuracy = cross_validate_single_dataset(
    "/path/to/MEG-Data",
    2,
    classifier="multiclass-svm",
)
```

If `PYMEGDEC_DATA_DIR` or `.pymegdec-data-dir` is configured, pass `None` for the
first argument:

```python
transfer_accuracy = evaluate_model_transfer(None, 2, classifier="multiclass-svm")
cv_accuracy = cross_validate_single_dataset(None, 2, classifier="multiclass-svm")
```

## Stimulus decoding

Use `StimulusDecodingConfig` to make time-resolved decoding parameters explicit:

```python
from pymegdec.stimulus_decoding import (
    StimulusDecodingConfig,
    export_time_resolved_stimulus_decoding,
    window_centers_from_range,
)

config = StimulusDecodingConfig(
    window_centers=window_centers_from_range((-0.2, 0.6), 0.05),
    window_size=0.1,
    null_window_center=float("nan"),
    classifier="multiclass-svm",
    components_pca=100,
)

rows, summary_rows = export_time_resolved_stimulus_decoding(
    data_folder="/path/to/MEG-Data",
    participants=[2],
    output_path="outputs/part2_stimulus_decoding.csv",
    summary_output_path="outputs/part2_stimulus_decoding_summary.csv",
    config=config,
)
```

## Data-directory resolver

Use `resolve_data_folder` when writing new workflows that need participant MAT
files:

```python
from pymegdec.data_config import resolve_data_folder

data_folder = resolve_data_folder(
    None,
    required=True,
    required_files=["Part2Data.mat", "Part2CueData.mat"],
)
```

Resolution order is documented in [Data configuration](data.md).

## Alpha metrics

Alpha metric command-line scripts share parsing helpers from `pymegdec.cli` and
write CSV rows through the alpha metrics utilities. For new code, keep the core
calculation in package modules and let top-level scripts handle command-line
argument parsing only.

## Classifiers

`pymegdec.classifiers.CLASSIFIER_REGISTRY` extends the NeuRepTrace classifier
registry with PyMEGDec-specific optional backends:

- `xgboost`, installed with the `xgboost` extra.
- `pytorch-mlp`, installed with the `torch` extra.

Use `pymegdec.classifiers.get_default_classifier_param(classifier)` when a
workflow wants the package default for a classifier.

## Module ownership

| Module                       | Responsibility                                                   |
|------------------------------|------------------------------------------------------------------|
| `data_config.py`             | Runtime data-directory resolution.                               |
| `preprocessing.py`           | Filtering, downsampling, window extraction, and PCA preparation. |
| `classifiers.py`             | Classifier registry and optional PyMEGDec backends.              |
| `cross_validation.py`        | Single-dataset participant cross-validation.                     |
| `model_transfer.py`          | Train-main / validate-cue transfer evaluation.                   |
| `stimulus_decoding.py`       | Time-resolved stimulus decoding and diagnostic summaries.        |
| `alpha_signal.py`            | Alpha filtering and phase extraction.                            |
| `alpha_metrics.py`           | Per-trial alpha power and phase-gradient metrics.                |
| `alpha_movement.py`          | Sensor-level alpha centroid trajectories.                        |
| `alpha_movement_analysis.py` | Pre/post movement summaries and plots.                           |
| `reaction_time_analysis.py`  | Alpha/RT joins and association summaries.                        |
