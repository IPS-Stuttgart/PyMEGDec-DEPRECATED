"""Cross-subject stimulus decoding smoke benchmarks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.alpha_signal import get_data_field
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from reptrace.decoding.windowed import fit_window_model as fit_reptrace_window_model
from reptrace.decoding.windowed import predict_window_model as predict_reptrace_window_model
from reptrace.metrics.confusion import confusion_counts, per_class_accuracy
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DEFAULT_CROSS_SUBJECT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_CROSS_SUBJECT_WINDOW_CENTER = 0.175
DEFAULT_CROSS_SUBJECT_WINDOW_SIZE = 0.1
DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW = (-0.5, 0.0)
DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES = 16
DEFAULT_CROSS_SUBJECT_FEATURE_MODE = "sensor_mean"
DEFAULT_CROSS_SUBJECT_NORMALIZATION = "subject_baseline_z"
DEFAULT_CROSS_SUBJECT_CLASSIFIER = "multiclass-svm"
DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA = 64
FEATURE_MODES = ("sensor_mean", "sensor_flat")
NORMALIZATION_MODES = ("none", "subject_z", "subject_baseline_z")


@dataclass(frozen=True)
class CrossSubjectStimulusConfig:
    """Parameters for the fixed-pipeline cross-subject stimulus smoke test."""

    window_center: float = DEFAULT_CROSS_SUBJECT_WINDOW_CENTER
    window_size: float = DEFAULT_CROSS_SUBJECT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str = DEFAULT_CROSS_SUBJECT_FEATURE_MODE
    normalization: str = DEFAULT_CROSS_SUBJECT_NORMALIZATION
    classifier: str = DEFAULT_CROSS_SUBJECT_CLASSIFIER
    classifier_param: object = float("nan")
    components_pca: int | float = DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA
    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES
    random_state: int | None = 0
    signflip_permutations: int = 10_000
    signflip_seed: int | None = 0


@dataclass(frozen=True)
class ParticipantFeatureSet:
    """Windowed features for one participant."""

    participant: int
    labels: np.ndarray
    features: np.ndarray
    baseline_features: np.ndarray | None
    n_channels: int
    n_window_samples: int
    n_baseline_samples: int


def evaluate_cross_subject_stimulus_smoke(data_folder, participants, *, config=None, progress=None):
    """Run fixed-pipeline leave-one-subject-out stimulus decoding on ``Part*Data.mat`` files only."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for a cross-subject smoke benchmark.")

    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    feature_sets = []
    for participant in participants:
        if progress is not None:
            progress(f"LOAD participant={participant}")
        feature_sets.append(load_participant_stimulus_features(data_folder, participant, config=config))

    outer_rows = []
    prediction_rows = []
    for test_participant in participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        train_sets = [feature_set for feature_set in feature_sets if feature_set.participant != test_participant]
        test_set = next(feature_set for feature_set in feature_sets if feature_set.participant == test_participant)
        outer_row, participant_predictions = _evaluate_outer_fold(
            train_sets,
            test_set,
            config=config,
            classifier_param=classifier_param,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")

    group_summary_rows = summarize_cross_subject_stimulus_smoke(outer_rows, config=config)
    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
    }


def load_participant_stimulus_features(data_folder, participant, *, config=None):
    """Load one participant's main ``Part*Data.mat`` file and extract fixed-window features."""

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    data_path = Path(resolve_data_folder(data_folder)) / f"Part{int(participant)}Data.mat"
    data = sio.loadmat(data_path)["data"][0]
    labels = _trialinfo_labels(data)
    features, n_window_samples = _extract_window_features(
        data,
        _centered_window(config.window_center, config.window_size),
        feature_mode=config.feature_mode,
    )
    baseline_features = None
    n_baseline_samples = 0
    if config.normalization == "subject_baseline_z":
        baseline_features, n_baseline_samples = _extract_window_features(data, config.baseline_window, feature_mode=config.feature_mode)
        if baseline_features.shape[1] != features.shape[1]:
            raise ValueError("subject_baseline_z requires baseline and decoding features with matching dimensions.")
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} labels but {features.shape[0]} feature rows.")
    return ParticipantFeatureSet(
        participant=int(participant),
        labels=labels,
        features=features,
        baseline_features=baseline_features,
        n_channels=int(_trial_signal(data, 0).shape[0]),
        n_window_samples=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
    )


def summarize_cross_subject_stimulus_smoke(outer_rows, *, config=None):
    """Summarize held-out participant scores with a one-sided subject-level sign-flip test."""

    if not outer_rows:
        return []

    config = _normalized_config(config or CrossSubjectStimulusConfig())
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    p_value = _one_sided_signflip_p_value(differences, n_permutations=config.signflip_permutations, seed=config.signflip_seed)
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "window_start_s": _centered_window(config.window_center, config.window_size)[0],
            "window_stop_s": _centered_window(config.window_center, config.window_size)[1],
            "baseline_window_start_s": config.baseline_window[0],
            "baseline_window_stop_s": config.baseline_window[1],
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "classifier": config.classifier,
            "components_pca": config.components_pca,
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "mean_above_chance": float(np.mean(differences)),
            "percent_above_chance": float(100.0 * np.mean(differences)),
            "participants_above_chance": int(np.sum(balanced > chance)),
            "participants_at_or_below_chance": int(np.sum(balanced <= chance)),
            "one_sided_signflip_p_value": p_value,
        }
    ]


def summarize_cross_subject_predictions(prediction_rows):
    """Return confusion-count and per-stimulus recall summaries for cross-subject predictions."""

    if not prediction_rows:
        return [], []

    import pandas as pd

    frame = pd.DataFrame(prediction_rows)
    group_columns = (
        "window_center_s",
        "feature_mode",
        "normalization",
        "classifier",
        "components_pca",
    )
    confusion = confusion_counts(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        group_columns=group_columns,
    )
    per_stimulus = per_class_accuracy(
        frame,
        true_column="true_stimulus",
        predicted_column="predicted_stimulus",
        participant_column="test_participant",
        group_columns=group_columns,
    )
    return confusion.to_dict(orient="records"), per_stimulus.to_dict(orient="records")


def export_cross_subject_stimulus_smoke(
    data_folder,
    participants,
    *,
    outer_output_path,
    group_summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    config=None,
    progress=None,
):
    """Run the cross-subject smoke benchmark and write compact CSV artifacts."""

    artifacts = evaluate_cross_subject_stimulus_smoke(data_folder, participants, config=config, progress=progress)
    write_alpha_metrics_csv(artifacts["outer"], outer_output_path)
    if group_summary_output_path:
        write_alpha_metrics_csv(artifacts["group_summary"], group_summary_output_path)
    if predictions_output_path:
        write_alpha_metrics_csv(artifacts["predictions"], predictions_output_path)
    if confusion_output_path:
        write_alpha_metrics_csv(artifacts["confusion"], confusion_output_path)
    if per_stimulus_output_path:
        write_alpha_metrics_csv(artifacts["per_stimulus"], per_stimulus_output_path)
    return artifacts


def _evaluate_outer_fold(train_sets, test_set, *, config, classifier_param):
    train_features = np.vstack([_normalized_subject_features(feature_set, config) for feature_set in train_sets])
    train_labels_one_based = np.concatenate([feature_set.labels for feature_set in train_sets])
    test_features = _normalized_subject_features(test_set, config)
    test_labels_one_based = test_set.labels
    train_labels = train_labels_one_based - 1
    test_labels = test_labels_one_based - 1

    train_window = _centered_window(config.window_center, config.window_size)
    model_bundle = fit_reptrace_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            config.classifier,
            classifier_param,
            random_state=config.random_state,
        ),
        components_pca=config.components_pca,
        train_window=train_window,
    )
    predictions, _scores = predict_reptrace_window_model(model_bundle, test_features)
    accuracy = float(accuracy_score(test_labels, predictions))
    balanced_accuracy = float(balanced_accuracy_score(test_labels, predictions))
    chance_accuracy = 1.0 / config.chance_classes
    train_class_counts = Counter(train_labels_one_based.tolist())
    test_class_counts = Counter(test_labels_one_based.tolist())
    train_participants = tuple(feature_set.participant for feature_set in train_sets)

    outer_row = {
        "outer_fold": int(test_set.participant),
        "test_participant": int(test_set.participant),
        "train_participants": ",".join(str(participant) for participant in train_participants),
        "n_train_participants": len(train_sets),
        "n_test_participants": 1,
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": train_window[0],
        "window_stop_s": train_window[1],
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced_accuracy,
        "balanced_percent": 100.0 * balanced_accuracy,
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "above_chance": bool(balanced_accuracy > chance_accuracy),
        "n_train_trials": int(train_labels.shape[0]),
        "n_test_trials": int(test_labels.shape[0]),
        "n_train_classes": int(len(train_class_counts)),
        "n_test_classes": int(len(test_class_counts)),
        "min_train_trials_per_class": int(min(train_class_counts.values())),
        "min_test_trials_per_class": int(min(test_class_counts.values())),
        "classifier": config.classifier,
        "classifier_param": classifier_param,
        "components_pca": config.components_pca,
        "actual_components_pca": model_bundle.actual_components_pca,
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "n_channels": test_set.n_channels,
        "n_window_samples": test_set.n_window_samples,
        "n_baseline_samples": test_set.n_baseline_samples,
    }
    prediction_rows = _prediction_rows(test_set, test_labels, predictions, config=config, actual_components_pca=model_bundle.actual_components_pca)
    return outer_row, prediction_rows


def _prediction_rows(test_set, test_labels, predictions, *, config, actual_components_pca):
    train_window = _centered_window(config.window_center, config.window_size)
    rows = []
    for trial_idx, (true_label, predicted_label) in enumerate(zip(test_labels, predictions)):
        true_stimulus = int(true_label) + 1
        predicted_stimulus = int(predicted_label) + 1
        rows.append(
            {
                "outer_fold": int(test_set.participant),
                "test_participant": int(test_set.participant),
                "window_center_s": config.window_center,
                "window_start_s": train_window[0],
                "window_stop_s": train_window[1],
                "feature_mode": config.feature_mode,
                "normalization": config.normalization,
                "classifier": config.classifier,
                "components_pca": config.components_pca,
                "actual_components_pca": actual_components_pca,
                "trial": int(trial_idx),
                "test_trial_index": int(trial_idx),
                "test_trial_number": int(trial_idx + 1),
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "correct": bool(predicted_label == true_label),
            }
        )
    return rows


def _extract_window_features(data, time_window, *, feature_mode):
    feature_mode = _normalize_feature_mode(feature_mode)
    time_vector = _time_vector(data, 0)
    mask = _time_mask(time_vector, time_window)
    features = []
    for trial_idx in range(_count_trials(data)):
        signal = _trial_signal(data, trial_idx)
        window_signal = signal[:, mask]
        if feature_mode == "sensor_mean":
            feature = np.mean(window_signal, axis=1)
        elif feature_mode == "sensor_flat":
            feature = window_signal.reshape(-1, order="F")
        else:
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")
        features.append(feature)
    return np.vstack(features), int(np.sum(mask))


def _trialinfo_labels(data):
    trialinfo = _unwrap_singleton(get_data_field(data, "trialinfo"))
    return np.asarray(trialinfo, dtype=int).ravel()


def _count_trials(data):
    trial_field = _unwrap_outer_cell(get_data_field(data, "trial"))
    values = np.asarray(trial_field, dtype=object)
    if values.ndim == 2 and values.shape[0] == 1:
        return int(values.shape[1])
    if values.ndim == 2 and values.shape[1] == 1:
        return int(values.shape[0])
    return int(values.size)


def _time_vector(data, trial_idx):
    return np.asarray(_cell_item(get_data_field(data, "time"), trial_idx), dtype=float).ravel()


def _trial_signal(data, trial_idx):
    return np.asarray(_cell_item(get_data_field(data, "trial"), trial_idx), dtype=float)


def _cell_item(cell, index):
    values = np.asarray(_unwrap_outer_cell(cell), dtype=object)
    if values.ndim == 0:
        return _unwrap_singleton(values.item())
    if values.ndim == 2 and values.shape[0] == 1:
        return _unwrap_singleton(values[0, index])
    if values.ndim == 2 and values.shape[1] == 1:
        return _unwrap_singleton(values[index, 0])
    return _unwrap_singleton(values[index])


def _unwrap_outer_cell(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _unwrap_singleton(value):
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _time_mask(time_vector, time_window):
    start, stop = time_window
    if start >= stop:
        raise ValueError("time_window start must be before stop.")
    tolerance = 1e-12
    mask = (time_vector >= start - tolerance) & (time_vector <= stop + tolerance)
    if not np.any(mask):
        raise ValueError(f"time_window {time_window} does not overlap the data.")
    return mask


def _normalized_subject_features(feature_set, config):
    if config.normalization == "none":
        return feature_set.features
    if config.normalization == "subject_z":
        reference = feature_set.features
    elif config.normalization == "subject_baseline_z":
        if feature_set.baseline_features is None:
            raise ValueError("subject_baseline_z requires baseline features.")
        reference = feature_set.baseline_features
    else:
        raise ValueError(f"Unsupported normalization: {config.normalization}")
    mean = np.mean(reference, axis=0, keepdims=True)
    std = np.std(reference, axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (feature_set.features - mean) / std


def _centered_window(center, size):
    return float(np.round(center - size / 2, 10)), float(np.round(center + size / 2, 10))


def _sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _one_sided_signflip_p_value(differences, *, n_permutations, seed):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan
    observed = float(np.mean(differences))
    if observed <= 0:
        return 1.0
    if differences.size <= 16:
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * differences.size)).T.reshape(-1, differences.size)
        null_means = signs @ differences / differences.size
        return float(np.mean(null_means >= observed))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), differences.size))
    null_means = signs @ differences / differences.size
    return float((np.sum(null_means >= observed) + 1) / (int(n_permutations) + 1))


def _normalized_config(config):
    return CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=_normalize_feature_mode(config.feature_mode),
        normalization=_normalize_normalization(config.normalization),
        classifier=config.classifier,
        classifier_param=config.classifier_param,
        components_pca=config.components_pca,
        chance_classes=config.chance_classes,
        random_state=config.random_state,
        signflip_permutations=config.signflip_permutations,
        signflip_seed=config.signflip_seed,
    )


def _normalize_feature_mode(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in FEATURE_MODES:
        raise ValueError(f"feature_mode must be one of {FEATURE_MODES}.")
    return normalized


def _normalize_normalization(value):
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_MODES}.")
    return normalized
