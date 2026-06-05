"""Strict source-inner compact+latent logit stacking for BUSH-MEG stimulus LOSO.

This command is intentionally slower than artifact-level ensembling.  For each
outer held-out participant it first creates source-inner out-of-fold predictions
from the compact logistic branch and the latent autoencoder branch, selects a
small scalar stacker on those source participants only, and then applies the
selected stacker to the untouched outer participant.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec import stimulus_latent_autoencoder as latent_ae
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec.stimulus_cross_subject import (
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    CrossSubjectStimulusConfig,
    make_cross_subject_candidate_configs,
    summarize_cross_subject_confusion_pairs,
    summarize_cross_subject_predictions,
)

DEFAULT_SOURCE_INNER_WINDOW_CENTER = 0.175
DEFAULT_SOURCE_INNER_WINDOW_SIZE = 0.150
DEFAULT_SOURCE_INNER_BASELINE_WINDOW = (-0.35, -0.05)
DEFAULT_SOURCE_INNER_COMPONENTS_PCA = 160
DEFAULT_SOURCE_INNER_LATENT_DIM = 32
DEFAULT_SOURCE_INNER_HIDDEN_DIM = 128
DEFAULT_SOURCE_INNER_RECONSTRUCTION_WEIGHT = 0.0
DEFAULT_SOURCE_INNER_COMPACT_CLASSIFIERS = ("multinomial-logistic", "multinomial-logistic-weighted")
DEFAULT_SOURCE_INNER_COMPACT_PARAMS = (0.3, 0.5)
DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))
DEFAULT_SOURCE_INNER_SCORE_MODE = "compact_logprob_latent_logit"
SOURCE_INNER_SCORE_MODES = (
    "compact_logprob_latent_logit",
    "compact_probability_latent_logit",
    "compact_probability_latent_probability",
    "compact_rank_latent_rank",
)
COMPACT_SCORE_NORMALIZATION_MODES = ("raw", "row_z_softmax", "rank_softmax")
SOURCE_INNER_CLASSIFIER = "source_inner_compact_latent_stack"


@dataclass(frozen=True)
class ScoreBlock:
    """Aligned per-trial class scores for one participant."""

    scores: np.ndarray
    labels: np.ndarray
    class_order: np.ndarray
    trial_indices: np.ndarray


@dataclass(frozen=True)
class SourceInnerStackConfig:
    """Configuration for strict source-inner compact+latent stacking."""

    compact_candidate_configs: tuple[CrossSubjectStimulusConfig, ...]
    latent_config: latent_ae.LatentAutoencoderConfig
    compact_score_normalization: str = "raw"
    stacker_score_mode: str = DEFAULT_SOURCE_INNER_SCORE_MODE
    stacker_weight_grid: tuple[float, ...] = DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID
    chance_classes: int = 16
    label_shuffle_control: bool = False
    label_shuffle_seed: int = 0


def _parse_float_sequence(value: str) -> tuple[float, ...]:
    return tuple(float(token.strip()) for token in value.replace(";", ",").split(",") if token.strip())


def _parse_token_sequence(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.replace(";", ",").split(",") if token.strip())


def _parse_time_window(value: str) -> tuple[float, float]:
    tokens = [token.strip() for token in value.replace(":", ",").split(",") if token.strip()]
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError("Expected a time window like -0.35,-0.05 or -0.35:-0.05.")
    start, stop = float(tokens[0]), float(tokens[1])
    if start > stop:
        raise argparse.ArgumentTypeError("Time-window start must be before stop.")
    return start, stop


def _parse_int_sequence(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple()
    return tuple(parse_participant_spec(value))


def _parse_classifier_params(value: str) -> tuple[float | str, ...]:
    params: list[float | str] = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "default":
            params.append(float("nan"))
        elif token.lower() == "auto-grid":
            params.append("auto-grid")
        else:
            params.append(float(token))
    return tuple(params)


def _normalize_stacker_weight_grid(values: Sequence[float] | str) -> tuple[float, ...]:
    if isinstance(values, str):
        values = _parse_float_sequence(values)
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("At least one stacker weight is required.")
    for weight in grid:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Stacker compact weights must be in [0, 1].")
    return tuple(dict.fromkeys(grid))


def _class_order(chance_classes: int) -> np.ndarray:
    return np.arange(1, int(chance_classes) + 1, dtype=int)


def _row_normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros_like(values)
    sums = np.sum(values, axis=1, keepdims=True)
    return np.divide(
        values,
        sums,
        out=np.full_like(values, 1.0 / values.shape[1]),
        where=sums > 1e-12,
    )


def _row_softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    output = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        finite = np.isfinite(row)
        if not np.any(finite):
            output[row_index] = np.full(row.shape[0], 1.0 / row.shape[0], dtype=float)
            continue
        sanitized = row.copy()
        sanitized[~finite] = np.min(sanitized[finite])
        logits = sanitized - np.max(sanitized)
        exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
        output[row_index] = exp_logits / np.sum(exp_logits)
    return output


def _row_z_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    centered = scores - np.nanmean(scores, axis=1, keepdims=True)
    scale = np.nanstd(centered, axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    ranks = np.empty_like(scores, dtype=float)
    for row_index, row in enumerate(scores):
        order = np.argsort(-np.where(np.isfinite(row), row, -np.inf), kind="mergesort")
        row_ranks = np.empty(row.shape[0], dtype=float)
        row_ranks[order] = np.arange(row.shape[0], dtype=float)
        ranks[row_index] = -row_ranks
    return ranks


def _probability_like_or_softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Scores must be a two-dimensional matrix.")
    sums = np.sum(scores, axis=1)
    probability_like = (
        np.all(np.isfinite(scores), axis=1)
        & np.all(scores >= 0.0, axis=1)
        & np.isclose(sums, 1.0, rtol=1e-3, atol=1e-6)
    )
    probabilities = np.zeros_like(scores, dtype=float)
    if np.any(probability_like):
        probabilities[probability_like] = scores[probability_like]
    if np.any(~probability_like):
        probabilities[~probability_like] = _row_softmax(scores[~probability_like])
    return _row_normalize_probabilities(probabilities)


def _align_columns(scores: np.ndarray, score_classes: np.ndarray, class_order: np.ndarray, *, fill_value: float | str = 0.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    score_classes = np.asarray(score_classes, dtype=int).ravel()
    if fill_value == "row_min":
        finite_scores = np.where(np.isfinite(scores), scores, np.nan)
        row_min = np.nanmin(finite_scores, axis=1)
        row_min = np.where(np.isfinite(row_min), row_min - 1.0, -1.0)
        aligned = np.repeat(row_min[:, None], class_order.shape[0], axis=1)
    else:
        aligned = np.full((scores.shape[0], class_order.shape[0]), float(fill_value), dtype=float)
    class_to_column = {int(label): column for column, label in enumerate(class_order.tolist())}
    for source_column, label in enumerate(score_classes.tolist()):
        target_column = class_to_column.get(int(label))
        if target_column is not None:
            aligned[:, target_column] = scores[:, source_column]
    return aligned


def _compact_score_probabilities(class_scores: np.ndarray, *, score_normalization: str) -> np.ndarray:
    mode = str(score_normalization).strip().lower().replace("-", "_")
    if mode == "raw":
        return _probability_like_or_softmax(class_scores)
    return cross_subject._class_score_probabilities(class_scores, score_normalization=mode)  # pylint: disable=protected-access


def _compact_score_block(
    *,
    feature_cache: dict,
    candidate_configs: Sequence[CrossSubjectStimulusConfig],
    train_participants: Sequence[int],
    test_participant: int,
    class_order: np.ndarray,
    compact_score_normalization: str,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 0,
    label_shuffle_context: Sequence[int] = (),
) -> ScoreBlock:
    matrices = []
    labels = None
    trial_indices = None
    for candidate_index, candidate_config in enumerate(candidate_configs, start=1):
        feature_sets = feature_cache[cross_subject._feature_cache_key(candidate_config)]  # pylint: disable=protected-access
        train_sets = [feature_sets[int(participant)] for participant in train_participants]
        test_set = feature_sets[int(test_participant)]
        fitted = cross_subject._fit_outer_fold_model(  # pylint: disable=protected-access
            train_sets,
            candidate_config,
            cross_subject._resolved_classifier_param(candidate_config),  # pylint: disable=protected-access
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(*tuple(int(value) for value in label_shuffle_context), candidate_index),
        )
        class_scores, score_classes = cross_subject._candidate_model_scores(fitted, test_set, candidate_config)  # pylint: disable=protected-access
        probabilities = _compact_score_probabilities(class_scores, score_normalization=compact_score_normalization)
        matrices.append(_align_columns(probabilities, np.asarray(score_classes, dtype=int) + 1, class_order, fill_value=0.0))
        labels = np.asarray(test_set.labels, dtype=int)
        trial_indices = np.asarray(
            getattr(test_set, "trial_indices", np.arange(labels.shape[0])),
            dtype=int,
        )
    if not matrices:
        raise ValueError("At least one compact candidate configuration is required.")
    compact_probabilities = _row_normalize_probabilities(np.mean(np.stack(matrices, axis=0), axis=0))
    return ScoreBlock(
        scores=compact_probabilities,
        labels=np.asarray(labels, dtype=int),
        class_order=class_order,
        trial_indices=np.asarray(trial_indices, dtype=int),
    )


def _latent_feature_config(config: latent_ae.LatentAutoencoderConfig) -> CrossSubjectStimulusConfig:
    return CrossSubjectStimulusConfig(
        window_center=config.window_center,
        window_size=config.window_size,
        baseline_window=config.baseline_window,
        feature_mode=config.feature_mode,
        normalization=config.normalization,
        alignment="none",
        classifier="latent_autoencoder",
        components_pca=config.components_pca,
        max_trials_per_class_per_participant=None,
        chance_classes=config.chance_classes,
        random_state=config.seed,
    )


def _load_latent_feature_sets(data_folder: Path, participants: Sequence[int], config: latent_ae.LatentAutoencoderConfig, *, progress=None) -> dict[int, object]:
    stimulus_config = _latent_feature_config(config)
    feature_sets = {}
    for participant in participants:
        if progress is not None:
            progress(f"LOAD latent_feature_set participant={participant}")
        feature_sets[int(participant)] = cross_subject.load_participant_stimulus_features(data_folder, participant, config=stimulus_config)
    return feature_sets


def _latent_score_block(  # pylint: disable=too-many-locals
    *,
    feature_sets: dict[int, object],
    train_participants: Sequence[int],
    test_participant: int,
    config: latent_ae.LatentAutoencoderConfig,
    class_order: np.ndarray,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 0,
    label_shuffle_context: Sequence[int] = (),
) -> ScoreBlock:
    train_participants = tuple(int(value) for value in train_participants)
    latent_ae._set_random_seeds(config.seed, num_threads=config.num_threads)  # pylint: disable=protected-access
    train_epoch_participants, validation_participants = latent_ae._split_source_participants(  # pylint: disable=protected-access
        train_participants,
        config.validation_source_count,
        strategy=config.validation_source_strategy,
        anchor=test_participant,
    )

    train_features_raw, train_labels_raw, train_subjects = latent_ae._concat_features(feature_sets, train_epoch_participants)  # pylint: disable=protected-access
    selected_epoch = int(config.epochs)
    if validation_participants:
        validation_features_raw, validation_labels_raw, validation_subjects = latent_ae._concat_features(feature_sets, validation_participants)  # pylint: disable=protected-access
        train_labels_epoch = np.asarray(train_labels_raw, dtype=int)
        validation_labels_epoch = np.asarray(validation_labels_raw, dtype=int)
        if label_shuffle_control:
            context = (*tuple(int(value) for value in label_shuffle_context), int(test_participant))
            train_labels_epoch = latent_ae._shuffle_labels_within_subjects(  # pylint: disable=protected-access
                train_labels_epoch,
                train_subjects,
                seed=label_shuffle_seed,
                context=(*context, 1),
            )
            validation_labels_epoch = latent_ae._shuffle_labels_within_subjects(  # pylint: disable=protected-access
                validation_labels_epoch,
                validation_subjects,
                seed=label_shuffle_seed,
                context=(*context, 2),
            )
        classes_epoch = np.asarray(sorted(set(int(value) for value in train_labels_epoch)), dtype=int)
        _pca, train_features_pca, validation_features_pca, _actual, _explained = latent_ae._fit_pca(  # pylint: disable=protected-access
            train_features_raw,
            validation_features_raw,
            components_pca=config.components_pca,
            seed=config.seed,
        )
        _model, fit_metadata = latent_ae._train_model(  # pylint: disable=protected-access
            train_features_pca,
            train_labels_epoch,
            train_subjects,
            classes=classes_epoch,
            subject_ids=train_epoch_participants,
            config=config,
            validation=(validation_features_pca, validation_labels_epoch),
        )
        selected_epoch = int(fit_metadata.get("best_epoch", config.epochs))

    final_train_features_raw, final_train_labels, final_train_subjects = latent_ae._concat_features(feature_sets, train_participants)  # pylint: disable=protected-access
    if label_shuffle_control:
        final_train_labels = latent_ae._shuffle_labels_within_subjects(  # pylint: disable=protected-access
            final_train_labels,
            final_train_subjects,
            seed=label_shuffle_seed,
            context=(*tuple(int(value) for value in label_shuffle_context), int(test_participant), 3),
        )
    test_features_raw, test_labels, _test_subjects = latent_ae._concat_features(feature_sets, (test_participant,))  # pylint: disable=protected-access
    classes = np.asarray(sorted(set(int(value) for value in final_train_labels)), dtype=int)
    _final_pca, final_train_pca, test_pca, _actual, _explained = latent_ae._fit_pca(  # pylint: disable=protected-access
        final_train_features_raw,
        test_features_raw,
        components_pca=config.components_pca,
        seed=config.seed,
    )
    device = latent_ae._resolve_device(config.device)  # pylint: disable=protected-access
    final_epochs = latent_ae._final_refit_epochs(selected_epoch, config) if config.refit_all_sources else int(config.epochs)  # pylint: disable=protected-access
    score_matrices = []
    for seed in latent_ae._effective_ensemble_seeds(config):  # pylint: disable=protected-access
        final_config = replace(config, seed=int(seed))
        final_model, _final_metadata = latent_ae._train_model(  # pylint: disable=protected-access
            final_train_pca,
            final_train_labels,
            final_train_subjects,
            classes=classes,
            subject_ids=train_participants,
            config=final_config,
            validation=None,
            max_epochs=final_epochs,
        )
        score_matrices.append(latent_ae._predict_scores(final_model, test_pca, device=device, batch_size=config.batch_size))  # pylint: disable=protected-access
    scores = np.mean(np.stack(score_matrices, axis=0), axis=0)
    trial_indices = np.asarray(
        getattr(feature_sets[int(test_participant)], "trial_indices", np.arange(test_labels.shape[0])),
        dtype=int,
    )
    return ScoreBlock(
        scores=_align_columns(scores, classes, class_order, fill_value="row_min"),
        labels=np.asarray(test_labels, dtype=int),
        class_order=class_order,
        trial_indices=trial_indices,
    )


def _validate_block_alignment(compact: ScoreBlock, latent: ScoreBlock) -> None:
    if not np.array_equal(compact.labels, latent.labels):
        raise ValueError("Compact and latent score blocks have different labels or trial order.")
    if not np.array_equal(compact.trial_indices, latent.trial_indices):
        raise ValueError("Compact and latent score blocks have different source trial indices.")
    if not np.array_equal(compact.class_order, latent.class_order):
        raise ValueError("Compact and latent score blocks use different class orders.")


def _stacker_source_scores(compact_probabilities: np.ndarray, latent_scores: np.ndarray, *, mode: str) -> tuple[np.ndarray, np.ndarray]:
    mode = str(mode).strip().lower().replace("-", "_")
    if mode == "compact_logprob_latent_logit":
        return np.log(np.clip(_row_normalize_probabilities(compact_probabilities), 1e-12, 1.0)), np.asarray(latent_scores, dtype=float)
    if mode == "compact_probability_latent_logit":
        return _row_normalize_probabilities(compact_probabilities), np.asarray(latent_scores, dtype=float)
    if mode == "compact_probability_latent_probability":
        return _row_normalize_probabilities(compact_probabilities), _row_softmax(latent_scores)
    if mode == "compact_rank_latent_rank":
        return _rank_scores(compact_probabilities), _rank_scores(latent_scores)
    raise ValueError(f"Unsupported source-inner stacker score mode: {mode}")


def _combine_sources(compact_probabilities: np.ndarray, latent_scores: np.ndarray, *, compact_weight: float, mode: str) -> np.ndarray:
    compact_source, latent_source = _stacker_source_scores(compact_probabilities, latent_scores, mode=mode)
    compact_weight = float(compact_weight)
    return compact_weight * compact_source + (1.0 - compact_weight) * latent_source


def _score_metrics(scores: np.ndarray, labels: np.ndarray, class_order: np.ndarray, *, chance_classes: int) -> dict:
    labels = np.asarray(labels, dtype=int)
    predicted = np.asarray(class_order, dtype=int)[np.argmax(scores, axis=1)]
    ranks = latent_ae._true_label_ranks(labels, scores, class_order)  # pylint: disable=protected-access
    finite = ranks[np.isfinite(ranks)]
    chance = 1.0 / float(chance_classes)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "percent": float(100.0 * accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "balanced_percent": float(100.0 * balanced_accuracy_score(labels, predicted)),
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top2_percent": float(100.0 * np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "top3_percent": float(100.0 * np.mean(ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite)) if finite.size else np.nan,
        "median_true_label_rank": float(np.median(finite)) if finite.size else np.nan,
        "chance_accuracy": chance,
        "chance_percent": 100.0 * chance,
        "top2_chance_accuracy": min(2.0 * chance, 1.0),
        "top2_chance_percent": min(200.0 * chance, 100.0),
        "top3_chance_accuracy": min(3.0 * chance, 1.0),
        "top3_chance_percent": min(300.0 * chance, 100.0),
        "chance_mean_rank": 0.5 * (float(chance_classes) + 1.0),
        "predicted_labels": predicted,
        "ranks": ranks,
    }


def _fit_scalar_stacker(
    compact_probabilities: np.ndarray,
    latent_scores: np.ndarray,
    labels: np.ndarray,
    class_order: np.ndarray,
    *,
    weight_grid: Sequence[float],
    mode: str,
    chance_classes: int,
) -> tuple[float, list[dict]]:
    rows = []
    for compact_weight in weight_grid:
        combined = _combine_sources(compact_probabilities, latent_scores, compact_weight=compact_weight, mode=mode)
        metrics = _score_metrics(combined, labels, class_order, chance_classes=chance_classes)
        rows.append(
            {
                "compact_weight": float(compact_weight),
                "latent_weight": float(1.0 - float(compact_weight)),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "balanced_percent": metrics["balanced_percent"],
                "accuracy": metrics["accuracy"],
                "percent": metrics["percent"],
                "top2_accuracy": metrics["top2_accuracy"],
                "top2_percent": metrics["top2_percent"],
                "top3_accuracy": metrics["top3_accuracy"],
                "top3_percent": metrics["top3_percent"],
                "mean_true_label_rank": metrics["mean_true_label_rank"],
            }
        )
    best = max(rows, key=lambda row: (row["balanced_accuracy"], row["top2_accuracy"], row["top3_accuracy"], -abs(row["compact_weight"] - 0.5)))
    return float(best["compact_weight"]), rows


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _format_counter(counter: Counter) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=lambda value: str(value)))


def _participant_list(values: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in values)


def _outer_row(
    *,
    test_participant: int,
    train_participants: Sequence[int],
    selected_weight: float,
    scores: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_order: np.ndarray,
    config: SourceInnerStackConfig,
    selected_inner: dict,
) -> dict:
    metrics = _score_metrics(scores, labels, class_order, chance_classes=config.chance_classes)
    return {
        "outer_fold": int(test_participant),
        "test_participant": int(test_participant),
        "train_participants": _participant_list(train_participants),
        "n_train_participants": int(len(train_participants)),
        "classifier": SOURCE_INNER_CLASSIFIER,
        "stacker": "scalar_weight_grid",
        "stacker_score_mode": config.stacker_score_mode,
        "compact_weight": float(selected_weight),
        "latent_weight": float(1.0 - selected_weight),
        "n_compact_candidates": int(len(config.compact_candidate_configs)),
        "compact_score_normalization": config.compact_score_normalization,
        "window_center_s": config.latent_config.window_center,
        "window_size_s": config.latent_config.window_size,
        "window_start_s": config.latent_config.window_center - 0.5 * config.latent_config.window_size,
        "window_stop_s": config.latent_config.window_center + 0.5 * config.latent_config.window_size,
        "baseline_window_start_s": config.latent_config.baseline_window[0],
        "baseline_window_stop_s": config.latent_config.baseline_window[1],
        "feature_mode": config.latent_config.feature_mode,
        "normalization": config.latent_config.normalization,
        "alignment": "none",
        "components_pca": config.latent_config.components_pca,
        "latent_dim": config.latent_config.latent_dim,
        "hidden_dim": config.latent_config.hidden_dim,
        "reconstruction_weight": config.latent_config.reconstruction_weight,
        "label_shuffle_control": config.label_shuffle_control,
        "label_shuffle_seed": config.label_shuffle_seed if config.label_shuffle_control else "",
        "n_test_trials": int(labels.shape[0]),
        "n_test_classes": int(np.unique(labels).shape[0]),
        "test_label_counts": _format_counter(Counter(labels.tolist())),
        "predicted_label_counts": _format_counter(Counter(predictions.tolist())),
        "selected_inner_balanced_accuracy": selected_inner["balanced_accuracy"],
        "selected_inner_top2_accuracy": selected_inner["top2_accuracy"],
        "selected_inner_top3_accuracy": selected_inner["top3_accuracy"],
        **{key: value for key, value in metrics.items() if key not in {"predicted_labels", "ranks"}},
        "above_chance": bool(metrics["balanced_accuracy"] > metrics["chance_accuracy"]),
    }


def _prediction_rows(
    *,
    test_participant: int,
    compact: ScoreBlock,
    latent: ScoreBlock,
    scores: np.ndarray,
    predictions: np.ndarray,
    selected_weight: float,
    config: SourceInnerStackConfig,
    extra_fields: dict | None = None,
) -> list[dict]:
    rows = []
    extra_fields = dict(extra_fields or {})
    ranks = latent_ae._true_label_ranks(compact.labels, scores, compact.class_order)  # pylint: disable=protected-access
    for row_index, true_label in enumerate(compact.labels.tolist()):
        row = {
            "test_participant": int(test_participant),
            "trial": int(row_index),
            "source_trial_index": int(compact.trial_indices[row_index]),
            "true_label": int(true_label),
            "predicted_label": int(predictions[row_index]),
            "true_stimulus": int(true_label),
            "predicted_stimulus": int(predictions[row_index]),
            "correct": bool(int(true_label) == int(predictions[row_index])),
            "true_label_rank": float(ranks[row_index]),
            "classifier": SOURCE_INNER_CLASSIFIER,
            "stacker_score_mode": config.stacker_score_mode,
            "compact_weight": float(selected_weight),
            "latent_weight": float(1.0 - selected_weight),
            "window_center_s": config.latent_config.window_center,
            "window_size_s": config.latent_config.window_size,
            "feature_mode": config.latent_config.feature_mode,
            "normalization": config.latent_config.normalization,
            "components_pca": config.latent_config.components_pca,
            "latent_dim": config.latent_config.latent_dim,
            "reconstruction_weight": config.latent_config.reconstruction_weight,
        }
        row.update(extra_fields)
        for class_label, value in zip(compact.class_order.tolist(), scores[row_index], strict=True):
            row[f"score_{int(class_label)}"] = float(value)
        for class_label, value in zip(compact.class_order.tolist(), compact.scores[row_index], strict=True):
            row[f"compact_probability_{int(class_label)}"] = float(value)
        for class_label, value in zip(compact.class_order.tolist(), latent.scores[row_index], strict=True):
            row[f"latent_logit_{int(class_label)}"] = float(value)
        rows.append(row)
    return rows


def _group_summary(outer_rows: list[dict], config: SourceInnerStackConfig) -> list[dict]:
    if not outer_rows:
        return []
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    accuracy = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = np.asarray([float(row["top2_accuracy"]) for row in outer_rows], dtype=float)
    top3 = np.asarray([float(row["top3_accuracy"]) for row in outer_rows], dtype=float)
    ranks = np.asarray([float(row["mean_true_label_rank"]) for row in outer_rows], dtype=float)
    chance = 1.0 / float(config.chance_classes)
    return [
        {
            "classifier": SOURCE_INNER_CLASSIFIER,
            "stacker": "scalar_weight_grid",
            "stacker_score_mode": config.stacker_score_mode,
            "compact_score_normalization": config.compact_score_normalization,
            "n_compact_candidates": int(len(config.compact_candidate_configs)),
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "window_center_s": config.latent_config.window_center,
            "window_size_s": config.latent_config.window_size,
            "feature_mode": config.latent_config.feature_mode,
            "normalization": config.latent_config.normalization,
            "components_pca": config.latent_config.components_pca,
            "latent_dim": config.latent_config.latent_dim,
            "hidden_dim": config.latent_config.hidden_dim,
            "reconstruction_weight": config.latent_config.reconstruction_weight,
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(accuracy)),
            "percent_mean": float(100.0 * np.mean(accuracy)),
            "accuracy_sem": _sem(accuracy),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "top2_accuracy_mean": float(np.mean(top2)),
            "top2_percent_mean": float(100.0 * np.mean(top2)),
            "top2_percent_sem": float(100.0 * _sem(top2)),
            "top3_accuracy_mean": float(np.mean(top3)),
            "top3_percent_mean": float(100.0 * np.mean(top3)),
            "top3_percent_sem": float(100.0 * _sem(top3)),
            "mean_true_label_rank_mean": float(np.mean(ranks)),
            "mean_true_label_rank_sem": _sem(ranks),
            "selected_compact_weight_counts": _format_counter(Counter(float(row["compact_weight"]) for row in outer_rows)),
            "participants_above_chance": int(np.sum(balanced > chance)),
            "participants_total": int(balanced.size),
            "label_shuffle_control": config.label_shuffle_control,
            "label_shuffle_seed": config.label_shuffle_seed if config.label_shuffle_control else "",
        }
    ]


def evaluate_source_inner_logit_stack(  # pylint: disable=too-many-locals
    data_folder,
    participants: Sequence[int],
    *,
    outer_participants: Sequence[int] | None = None,
    config: SourceInnerStackConfig,
    progress=None,
) -> dict[str, list[dict]]:
    """Run strict LOSO source-inner scalar stacking."""

    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(value) for value in participants)
    outer_participants = tuple(int(value) for value in (outer_participants if outer_participants is not None else participants))
    unknown_outer = sorted(set(outer_participants) - set(participants))
    if unknown_outer:
        raise ValueError(f"Outer participants must be included in participants: {unknown_outer}")
    if len(participants) < 4:
        raise ValueError("At least four participants are required for source-inner stacking.")

    class_order = _class_order(config.chance_classes)
    compact_feature_cache = cross_subject._load_feature_cache(  # pylint: disable=protected-access
        data_folder,
        participants,
        config.compact_candidate_configs,
        progress=progress,
    )
    latent_feature_sets = _load_latent_feature_sets(data_folder, participants, config.latent_config, progress=progress)
    outer_rows: list[dict] = []
    inner_rows: list[dict] = []
    selected_rows: list[dict] = []
    prediction_rows: list[dict] = []
    source_inner_prediction_rows: list[dict] = []

    for outer_index, test_participant in enumerate(outer_participants, start=1):
        if progress is not None:
            progress(f"START source_inner_stack outer_test_participant={test_participant} outer_index={outer_index}/{len(outer_participants)}")
        outer_train = tuple(participant for participant in participants if participant != test_participant)
        inner_compact = []
        inner_latent = []
        inner_labels = []
        inner_blocks = []
        for validation_participant in outer_train:
            inner_train = tuple(participant for participant in outer_train if participant != validation_participant)
            if progress is not None:
                progress(f"START stacker_inner outer={test_participant} validation={validation_participant}")
            compact_block = _compact_score_block(
                feature_cache=compact_feature_cache,
                candidate_configs=config.compact_candidate_configs,
                train_participants=inner_train,
                test_participant=validation_participant,
                class_order=class_order,
                compact_score_normalization=config.compact_score_normalization,
                label_shuffle_control=config.label_shuffle_control,
                label_shuffle_seed=config.label_shuffle_seed,
                label_shuffle_context=(test_participant, validation_participant),
            )
            latent_block = _latent_score_block(
                feature_sets=latent_feature_sets,
                train_participants=inner_train,
                test_participant=validation_participant,
                config=config.latent_config,
                class_order=class_order,
                label_shuffle_control=config.label_shuffle_control,
                label_shuffle_seed=config.label_shuffle_seed,
                label_shuffle_context=(test_participant, validation_participant),
            )
            _validate_block_alignment(compact_block, latent_block)
            inner_compact.append(compact_block.scores)
            inner_latent.append(latent_block.scores)
            inner_labels.append(compact_block.labels)
            inner_blocks.append((int(validation_participant), compact_block, latent_block))
        inner_compact_scores = np.vstack(inner_compact)
        inner_latent_scores = np.vstack(inner_latent)
        inner_label_vector = np.concatenate(inner_labels)
        selected_weight, grid_rows = _fit_scalar_stacker(
            inner_compact_scores,
            inner_latent_scores,
            inner_label_vector,
            class_order,
            weight_grid=config.stacker_weight_grid,
            mode=config.stacker_score_mode,
            chance_classes=config.chance_classes,
        )
        for grid_row in grid_rows:
            grid_row.update(
                {
                    "outer_fold": int(test_participant),
                    "test_participant": int(test_participant),
                    "selection_mode": "source_inner_loso",
                    "stacker_score_mode": config.stacker_score_mode,
                    "n_inner_trials": int(inner_label_vector.shape[0]),
                    "n_inner_participants": int(len(outer_train)),
                }
            )
            inner_rows.append(grid_row)
        selected_inner = max((row for row in grid_rows if float(row["compact_weight"]) == selected_weight), key=lambda row: row["balanced_accuracy"])
        selected_row = {
            "outer_fold": int(test_participant),
            "test_participant": int(test_participant),
            "selection_mode": "source_inner_loso",
            "stacker": "scalar_weight_grid",
            "stacker_score_mode": config.stacker_score_mode,
            "selected_compact_weight": float(selected_weight),
            "selected_latent_weight": float(1.0 - selected_weight),
            "selected_inner_balanced_accuracy": selected_inner["balanced_accuracy"],
            "selected_inner_top2_accuracy": selected_inner["top2_accuracy"],
            "selected_inner_top3_accuracy": selected_inner["top3_accuracy"],
            "selected_inner_mean_true_label_rank": selected_inner["mean_true_label_rank"],
            "n_inner_participants": int(len(outer_train)),
            "n_inner_trials": int(inner_label_vector.shape[0]),
        }
        selected_rows.append(selected_row)
        for validation_participant, compact_block, latent_block in inner_blocks:
            inner_scores = _combine_sources(compact_block.scores, latent_block.scores, compact_weight=selected_weight, mode=config.stacker_score_mode)
            inner_predictions = class_order[np.argmax(inner_scores, axis=1)]
            source_inner_prediction_rows.extend(
                _prediction_rows(
                    test_participant=validation_participant,
                    compact=compact_block,
                    latent=latent_block,
                    scores=inner_scores,
                    predictions=inner_predictions,
                    selected_weight=selected_weight,
                    config=config,
                    extra_fields={
                        "prediction_role": "source_inner_validation",
                        "outer_test_participant": int(test_participant),
                        "inner_validation_participant": int(validation_participant),
                    },
                )
            )

        compact_outer = _compact_score_block(
            feature_cache=compact_feature_cache,
            candidate_configs=config.compact_candidate_configs,
            train_participants=outer_train,
            test_participant=test_participant,
            class_order=class_order,
            compact_score_normalization=config.compact_score_normalization,
            label_shuffle_control=config.label_shuffle_control,
            label_shuffle_seed=config.label_shuffle_seed,
            label_shuffle_context=(test_participant, 0),
        )
        latent_outer = _latent_score_block(
            feature_sets=latent_feature_sets,
            train_participants=outer_train,
            test_participant=test_participant,
            config=config.latent_config,
            class_order=class_order,
            label_shuffle_control=config.label_shuffle_control,
            label_shuffle_seed=config.label_shuffle_seed,
            label_shuffle_context=(test_participant, 0),
        )
        _validate_block_alignment(compact_outer, latent_outer)
        final_scores = _combine_sources(compact_outer.scores, latent_outer.scores, compact_weight=selected_weight, mode=config.stacker_score_mode)
        predictions = class_order[np.argmax(final_scores, axis=1)]
        outer_row = _outer_row(
            test_participant=test_participant,
            train_participants=outer_train,
            selected_weight=selected_weight,
            scores=final_scores,
            labels=compact_outer.labels,
            predictions=predictions,
            class_order=class_order,
            config=config,
            selected_inner=selected_inner,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(
            _prediction_rows(
                test_participant=test_participant,
                compact=compact_outer,
                latent=latent_outer,
                scores=final_scores,
                predictions=predictions,
                selected_weight=selected_weight,
                config=config,
                extra_fields={
                    "prediction_role": "outer_test",
                    "outer_test_participant": int(test_participant),
                    "inner_validation_participant": "",
                },
            )
        )
        if progress is not None:
            progress(
                "DONE source_inner_stack "
                f"outer_test_participant={test_participant} "
                f"compact_weight={selected_weight:.3f} "
                f"inner_balanced={selected_inner['balanced_accuracy']:.4f} "
                f"outer_balanced={outer_row['balanced_accuracy']:.4f}"
            )

    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    return {
        "outer": outer_rows,
        "inner_validation": inner_rows,
        "selected": selected_rows,
        "predictions": prediction_rows,
        "source_inner_predictions": source_inner_prediction_rows,
        "group_summary": _group_summary(outer_rows, config),
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": summarize_cross_subject_confusion_pairs(prediction_rows),
    }


def export_source_inner_logit_stack(  # pylint: disable=too-many-arguments
    data_folder,
    participants: Sequence[int],
    *,
    outer_participants: Sequence[int] | None = None,
    config: SourceInnerStackConfig,
    outer_output_path,
    group_summary_output_path=None,
    inner_validation_output_path=None,
    selected_output_path=None,
    predictions_output_path=None,
    source_inner_predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    progress=None,
) -> dict[str, list[dict]]:
    artifacts = evaluate_source_inner_logit_stack(
        data_folder,
        participants,
        outer_participants=outer_participants,
        config=config,
        progress=progress,
    )
    write_alpha_metrics_csv(artifacts["outer"], outer_output_path)
    if group_summary_output_path:
        write_alpha_metrics_csv(artifacts["group_summary"], group_summary_output_path)
    if inner_validation_output_path:
        write_alpha_metrics_csv(artifacts["inner_validation"], inner_validation_output_path)
    if selected_output_path:
        write_alpha_metrics_csv(artifacts["selected"], selected_output_path)
    if predictions_output_path:
        write_alpha_metrics_csv(artifacts["predictions"], predictions_output_path)
    if source_inner_predictions_output_path:
        write_alpha_metrics_csv(artifacts["source_inner_predictions"], source_inner_predictions_output_path)
    if confusion_output_path:
        write_alpha_metrics_csv(artifacts["confusion"], confusion_output_path)
    if per_stimulus_output_path:
        write_alpha_metrics_csv(artifacts["per_stimulus"], per_stimulus_output_path)
    if confusion_pairs_output_path:
        write_alpha_metrics_csv(artifacts["confusion_pairs"], confusion_pairs_output_path)
    return artifacts


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Strict source-inner compact+latent logit stacker for BUSH-MEG stimulus LOSO.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_CROSS_SUBJECT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participants to evaluate; defaults to all participants.")
    parser.add_argument("--window-center", type=float, default=DEFAULT_SOURCE_INNER_WINDOW_CENTER)
    parser.add_argument("--window-size", type=float, default=DEFAULT_SOURCE_INNER_WINDOW_SIZE)
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_SOURCE_INNER_BASELINE_WINDOW)
    parser.add_argument("--feature-mode", default="sensor_flat")
    parser.add_argument("--normalization", default="subject_baseline_whiten")
    parser.add_argument("--compact-classifiers", type=_parse_token_sequence, default=DEFAULT_SOURCE_INNER_COMPACT_CLASSIFIERS)
    parser.add_argument("--compact-classifier-params", type=_parse_classifier_params, default=DEFAULT_SOURCE_INNER_COMPACT_PARAMS)
    parser.add_argument("--components-pca", type=int, default=DEFAULT_SOURCE_INNER_COMPONENTS_PCA)
    parser.add_argument("--compact-score-normalization", choices=COMPACT_SCORE_NORMALIZATION_MODES, default="raw")
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_SOURCE_INNER_LATENT_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_SOURCE_INNER_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--reconstruction-weight", type=float, default=DEFAULT_SOURCE_INNER_RECONSTRUCTION_WEIGHT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-source-count", type=int, default=2)
    parser.add_argument("--validation-source-strategy", choices=("tail", "head", "spread", "rotating"), default=latent_ae.DEFAULT_LATENT_VALIDATION_SOURCE_STRATEGY)
    parser.add_argument("--validation-selection-metric", choices=("balanced_accuracy", "balanced_top2_top3_rank", "balanced_top2_top3_rank_balance"), default="balanced_accuracy")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--refit-all-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--final-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--final-min-epochs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-seeds", type=_parse_int_sequence, default=())
    parser.add_argument("--stacker-score-mode", choices=SOURCE_INNER_SCORE_MODES, default=DEFAULT_SOURCE_INNER_SCORE_MODE)
    parser.add_argument("--stacker-weight-grid", type=_normalize_stacker_weight_grid, default=DEFAULT_SOURCE_INNER_STACKER_WEIGHT_GRID)
    parser.add_argument("--chance-classes", type=int, default=16)
    parser.add_argument("--label-shuffle-control", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--outer-output", default="outputs/source_inner_stack_outer.csv")
    parser.add_argument("--summary-output", default="outputs/source_inner_stack_group_summary.csv")
    parser.add_argument("--inner-validation-output", default="outputs/source_inner_stack_inner_validation.csv")
    parser.add_argument("--selected-output", default="outputs/source_inner_stack_selected.csv")
    parser.add_argument("--predictions-output", default="outputs/source_inner_stack_predictions.csv")
    parser.add_argument("--source-inner-predictions-output", default="outputs/source_inner_stack_source_inner_predictions.csv")
    parser.add_argument("--confusion-output", default="outputs/source_inner_stack_confusion.csv")
    parser.add_argument("--per-stimulus-output", default="outputs/source_inner_stack_per_stimulus.csv")
    parser.add_argument("--confusion-pairs-output", default="outputs/source_inner_stack_confusion_pairs.csv")
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog)
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = parse_participant_spec(args.participants)
    if len(participants) < 4:
        parser.error("At least four participants are required.")
    outer_participants = parse_participant_spec(args.outer_participants) if args.outer_participants else None
    compact_candidate_configs = make_cross_subject_candidate_configs(
        window_centers=(args.window_center,),
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_modes=(args.feature_mode,),
        normalizations=(args.normalization,),
        alignments=("none",),
        classifiers=args.compact_classifiers,
        classifier_params=args.compact_classifier_params,
        components_pca_values=(args.components_pca,),
        chance_classes=args.chance_classes,
        random_state=args.seed,
    )
    latent_config = latent_ae.LatentAutoencoderConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        components_pca=args.components_pca,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        reconstruction_weight=args.reconstruction_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_source_count=args.validation_source_count,
        validation_source_strategy=args.validation_source_strategy,
        validation_selection_metric=args.validation_selection_metric,
        patience=args.patience,
        refit_all_sources=args.refit_all_sources,
        final_epoch_multiplier=args.final_epoch_multiplier,
        final_min_epochs=args.final_min_epochs,
        seed=args.seed,
        ensemble_seeds=tuple(args.ensemble_seeds),
        chance_classes=args.chance_classes,
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
        device=args.device,
        num_threads=args.num_threads,
    )
    config = SourceInnerStackConfig(
        compact_candidate_configs=compact_candidate_configs,
        latent_config=latent_config,
        compact_score_normalization=args.compact_score_normalization,
        stacker_score_mode=args.stacker_score_mode,
        stacker_weight_grid=tuple(args.stacker_weight_grid),
        chance_classes=args.chance_classes,
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    artifacts = export_source_inner_logit_stack(
        data_folder,
        participants,
        outer_participants=outer_participants,
        config=config,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        inner_validation_output_path=args.inner_validation_output,
        selected_output_path=args.selected_output,
        predictions_output_path=args.predictions_output,
        source_inner_predictions_output_path=args.source_inner_predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(artifacts['outer'])} untouched outer participant rows to {args.outer_output}")
    print(f"Wrote {len(artifacts['inner_validation'])} source-inner stacker rows to {args.inner_validation_output}")
    print(f"Wrote {len(artifacts['selected'])} selected stacker rows to {args.selected_output}")
    print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['predictions'])} trial prediction rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['source_inner_predictions'])} source-inner prediction rows to {args.source_inner_predictions_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
