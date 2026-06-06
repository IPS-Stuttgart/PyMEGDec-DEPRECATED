"""Experimental source-only latent autoencoder for BUSH-MEG stimulus LOSO decoding.

This module intentionally stays separate from the production nested-matrix logistic
pipeline.  It evaluates a fixed, source-only architecture:

    subject_baseline_whiten/windowed sensor features -> source-fitted PCA
    -> shared encoder -> classifier
                       -> subject-specific reconstruction decoder

No held-out-subject main-task trials are used for training or fitting PCA.
"""

from __future__ import annotations

import argparse
import copy
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec
from pymegdec.stimulus_cross_subject import (
    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
    CrossSubjectStimulusConfig,
    load_participant_stimulus_features,
    summarize_cross_subject_confusion_pairs,
    summarize_cross_subject_predictions,
)

DEFAULT_LATENT_PARTICIPANTS = DEFAULT_CROSS_SUBJECT_PARTICIPANTS
DEFAULT_LATENT_BASELINE_WINDOW = (-0.35, -0.05)
DEFAULT_LATENT_WINDOW_CENTER = 0.175
DEFAULT_LATENT_WINDOW_SIZE = 0.150
DEFAULT_LATENT_COMPONENTS_PCA = 160
DEFAULT_LATENT_DIM = 64
DEFAULT_LATENT_HIDDEN_DIM = 128
DEFAULT_LATENT_RECONSTRUCTION_WEIGHT = 0.03
DEFAULT_LATENT_TRAINING_PRESET = "none"
LATENT_TRAINING_PRESET_CHOICES = (
    "none",
    "anti_collapse_train",
    "anti_collapse_calibrated",
    "anti_collapse_refit",
    "anti_collapse_head_refit",
    "anti_collapse_head_blend",
    "anti_collapse_contrastive",
)
DEFAULT_LATENT_VALIDATION_SOURCE_STRATEGY = "rotating"
DEFAULT_LATENT_SELECTED_SCORE_CALIBRATION_CANDIDATES = (
    "none",
    "validation_argmax_class_bias_guarded",
    "validation_temperature_argmax_class_bias_guarded",
    "validation_rank_prior_bias_guarded",
    "validation_class_zscore_guarded",
    "validation_score_standardize_guarded",
    "validation_vector_bias_guarded",
    "validation_logistic_stack_guarded",
)
LATENT_HEAD_REFIT_CHOICES = (
    "none",
    "source_logistic",
    "validation_selected_source_logistic",
    "validation_selected_source_logistic_blend",
)


@dataclass(frozen=True)
class LatentAutoencoderConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for fixed source-only latent autoencoder LOSO decoding."""

    window_center: float = DEFAULT_LATENT_WINDOW_CENTER
    window_size: float = DEFAULT_LATENT_WINDOW_SIZE
    training_preset: str = DEFAULT_LATENT_TRAINING_PRESET
    baseline_window: tuple[float, float] = DEFAULT_LATENT_BASELINE_WINDOW
    feature_mode: str = "sensor_flat"
    normalization: str = "subject_baseline_whiten"
    components_pca: int = DEFAULT_LATENT_COMPONENTS_PCA
    latent_dim: int = DEFAULT_LATENT_DIM
    hidden_dim: int = DEFAULT_LATENT_HIDDEN_DIM
    dropout: float = 0.10
    input_dropout: float = 0.0
    reconstruction_weight: float = DEFAULT_LATENT_RECONSTRUCTION_WEIGHT
    subject_adversary_weight: float = 0.0
    prediction_balance_weight: float = 0.0
    prediction_balance_target_smoothing: float = 1.0
    prediction_balance_temperature: float = 1.0
    logit_mean_center_weight: float = 0.0
    class_bias_l2_weight: float = 0.0
    confidence_penalty_weight: float = 0.0
    label_smoothing: float = 0.0
    focal_loss_gamma: float = 0.0
    margin_loss_weight: float = 0.0
    margin_loss_value: float = 1.0
    soft_macro_recall_weight: float = 0.0
    soft_worst_class_recall_weight: float = 0.0
    supervised_contrastive_weight: float = 0.0
    supervised_contrastive_temperature: float = 0.20
    balanced_batch_sampling: bool = False
    subject_class_balanced_batch_sampling: bool = False
    validation_prediction_balance_weight: float = 0.0
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_source_count: int = 2
    validation_source_strategy: str = DEFAULT_LATENT_VALIDATION_SOURCE_STRATEGY
    validation_selection_metric: str = "balanced_accuracy"
    patience: int = 12
    refit_all_sources: bool = True
    final_epoch_multiplier: float = 1.0
    final_min_epochs: int = 0
    seed: int = 0
    ensemble_seeds: tuple[int, ...] = ()
    chance_classes: int = 16
    label_shuffle_control: bool = False
    label_shuffle_seed: int = 0
    latent_head_refit: str = "none"
    latent_head_refit_c_values: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)
    latent_head_refit_blend_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    latent_head_refit_selection_metric: str = "balanced_accuracy"
    score_calibration: str = "none"
    score_calibration_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    score_calibration_temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    score_calibration_logistic_c_values: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
    score_calibration_smoothing: float = 1.0
    score_calibration_confusion_smoothing: float = 4.0
    score_calibration_selection_metric: str = "balanced_accuracy"
    score_calibration_guard_tolerance: float = 0.0
    score_calibration_vector_steps: tuple[float, ...] = (0.5, 0.25, 0.125)
    score_calibration_vector_rounds: int = 2
    score_calibration_vector_l2: float = 0.0
    score_calibration_final_refit: bool = False
    prediction_postprocessing: str = "none"
    prediction_postprocessing_guard_tolerance: float = 0.0
    prediction_postprocessing_selection_metric: str = "balanced_accuracy"
    prediction_postprocessing_quota_strength: float = 1.0
    prediction_postprocessing_shrinkage_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    prediction_postprocessing_margin_thresholds: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
    device: str = "auto"
    num_threads: int = 1


def _lazy_torch():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The latent autoencoder experiment requires torch. Install with `pip install -e '.[torch]'` "
            "or use the workflow that installs the torch optional dependency."
        ) from exc
    return torch, nn, F


def _centered_window(center: float, size: float) -> tuple[float, float]:
    half = 0.5 * float(size)
    return float(center) - half, float(center) + half


def _parse_time_window(value: str) -> tuple[float, float]:
    tokens = [token.strip() for token in value.replace(":", ",").split(",") if token.strip()]
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError("Expected a time window like -0.35,-0.05 or -0.35:-0.05.")
    start, stop = (float(tokens[0]), float(tokens[1]))
    if start > stop:
        raise argparse.ArgumentTypeError("Time-window start must be before stop.")
    return start, stop


def _parse_float_sequence(value: str) -> tuple[float, ...]:
    """Parse a comma/semicolon separated float sequence."""

    return tuple(float(token.strip()) for token in value.replace(";", ",").split(",") if token.strip())


def _parse_int_sequence(value: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse a comma/semicolon separated int sequence."""

    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(int(token.strip()) for token in value.replace(";", ",").split(",") if token.strip())
    return tuple(int(token) for token in value)


def _parse_participants(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(parse_participant_spec(DEFAULT_LATENT_PARTICIPANTS))
    return tuple(parse_participant_spec(value))


def _effective_ensemble_seeds(config: LatentAutoencoderConfig) -> tuple[int, ...]:
    """Return deduplicated final-score ensemble seeds, defaulting to config.seed."""

    seeds = tuple(int(seed) for seed in config.ensemble_seeds)
    if not seeds:
        seeds = (int(config.seed),)
    return tuple(dict.fromkeys(seeds))


def _format_seed_sequence(seeds: Sequence[int]) -> str:
    return ";".join(str(int(seed)) for seed in seeds)


def _min_positive_temperature(value: float, target: float) -> float:
    value = float(value)
    target = float(target)
    if value <= 0.0:
        return target
    return min(value, target)


def _apply_latent_training_preset(config: LatentAutoencoderConfig, preset: str) -> LatentAutoencoderConfig:
    """Return ``config`` with a named, source-only latent-AE preset applied.

    The first latent smoke fold was above chance but showed hard prediction
    collapse: several classes had zero recall while a few classes absorbed many
    argmax predictions.  All knobs used here already operate on source training
    or source-validation data only; the preset simply makes the most relevant
    anti-collapse combination reproducible without long, error-prone CLI calls.

    ``anti_collapse_train`` changes training/epoch-selection only.
    ``anti_collapse_calibrated`` additionally enables guarded source-validation
    score calibration and source-validation-selected balanced assignment.
    ``anti_collapse_refit`` also replaces the joint neural classifier head with
    a source-validation-selected multinomial logistic probe on the frozen shared
    latent space.  This targets the smoke-run failure mode where the learned
    representation ranks classes reasonably but the neural head collapses to a few argmax classes.
    ``anti_collapse_head_refit`` adds a source-validation-selected logistic
    classifier on the learned latent space before the guarded calibration stack.
    ``anti_collapse_head_blend`` uses the same frozen-latent logistic probe, but
    source-validation-selects a convex blend between the neural head and the
    logistic probe instead of always replacing the neural head.
    ``anti_collapse_contrastive`` keeps the anti-collapse training safeguards but
    also turns on a small supervised contrastive latent loss.  This directly
    encourages same-stimulus trials from different source subjects to share latent
    neighborhoods before the classifier head is fitted.
    """

    preset = str(preset or DEFAULT_LATENT_TRAINING_PRESET)
    if preset not in LATENT_TRAINING_PRESET_CHOICES:
        raise ValueError(
            "latent training preset must be one of: "
            + ", ".join(LATENT_TRAINING_PRESET_CHOICES)
        )
    if preset == "none":
        return replace(config, training_preset="none")

    # Keep every epoch class-diverse and add gentle source-only objectives that
    # specifically penalize the collapse pattern seen in the smoke run.
    label_smoothing = max(float(config.label_smoothing), 0.05)
    input_dropout = max(float(config.input_dropout), 0.05)
    prediction_balance_weight = max(float(config.prediction_balance_weight), 0.02)
    prediction_balance_temperature = _min_positive_temperature(
        config.prediction_balance_temperature,
        0.10,
    )
    logit_mean_center_weight = max(float(config.logit_mean_center_weight), 0.003)
    class_bias_l2_weight = max(float(config.class_bias_l2_weight), 0.003)
    soft_macro_recall_weight = max(float(config.soft_macro_recall_weight), 0.02)
    # The smoke fold selected epoch 3 from only two validation sources.  A
    # slightly larger rotating validation set and a final minimum epoch count
    # should reduce early-stopping variance without touching held-out labels.
    validation_source_count = max(int(config.validation_source_count), 4)
    validation_prediction_balance_weight = max(
        float(config.validation_prediction_balance_weight),
        0.03,
    )
    final_min_epochs = max(int(config.final_min_epochs), 8)
    if preset == "anti_collapse_train":
        return replace(
            config,
            training_preset=preset,
            balanced_batch_sampling=True,
            subject_class_balanced_batch_sampling=True,
            label_smoothing=label_smoothing,
            input_dropout=input_dropout,
            prediction_balance_weight=prediction_balance_weight,
            prediction_balance_target_smoothing=1.0,
            prediction_balance_temperature=prediction_balance_temperature,
            logit_mean_center_weight=logit_mean_center_weight,
            class_bias_l2_weight=class_bias_l2_weight,
            soft_macro_recall_weight=soft_macro_recall_weight,
            validation_source_count=validation_source_count,
            validation_prediction_balance_weight=validation_prediction_balance_weight,
            validation_selection_metric="balanced_top2_top3_rank_balance",
            final_min_epochs=final_min_epochs,
        )

    if preset == "anti_collapse_refit":
        return replace(
            config,
            training_preset=preset,
            balanced_batch_sampling=True,
            label_smoothing=label_smoothing,
            input_dropout=input_dropout,
            prediction_balance_weight=prediction_balance_weight,
            prediction_balance_target_smoothing=1.0,
            prediction_balance_temperature=prediction_balance_temperature,
            logit_mean_center_weight=logit_mean_center_weight,
            class_bias_l2_weight=class_bias_l2_weight,
            soft_macro_recall_weight=soft_macro_recall_weight,
            validation_source_count=validation_source_count,
            validation_prediction_balance_weight=validation_prediction_balance_weight,
            validation_selection_metric="balanced_top2_top3_rank_balance",
            final_min_epochs=final_min_epochs,
            # The neural classifier head is trained jointly with reconstruction.
            # In the smoke run, the latent space still carried useful rank signal
            # while the argmax distribution collapsed.  A source-only logistic
            # probe on frozen latents gives the classifier a convex balanced
            # objective after representation learning, without touching held-out
            # main-task labels.
            latent_head_refit="validation_selected_source_logistic",
            latent_head_refit_selection_metric="balanced_top2_top3_rank_balance",
            score_calibration="validation_selected_guarded",
            score_calibration_selection_metric="balanced_top2_top3_rank_balance",
            score_calibration_guard_tolerance=min(float(config.score_calibration_guard_tolerance), 0.0),
            score_calibration_final_refit=True,
            prediction_postprocessing="validation_selected_balanced_assignment",
            prediction_postprocessing_selection_metric="balanced_top2_top3_rank_balance",
            prediction_postprocessing_guard_tolerance=min(
                float(config.prediction_postprocessing_guard_tolerance),
                0.0,
            ),
        )

    if preset == "anti_collapse_contrastive":
        return replace(
            config,
            training_preset=preset,
            balanced_batch_sampling=True,
            subject_class_balanced_batch_sampling=True,
            label_smoothing=label_smoothing,
            prediction_balance_weight=prediction_balance_weight,
            prediction_balance_target_smoothing=1.0,
            prediction_balance_temperature=prediction_balance_temperature,
            logit_mean_center_weight=logit_mean_center_weight,
            class_bias_l2_weight=class_bias_l2_weight,
            soft_macro_recall_weight=soft_macro_recall_weight,
            validation_source_count=validation_source_count,
            validation_prediction_balance_weight=validation_prediction_balance_weight,
            validation_selection_metric="balanced_top2_top3_rank_balance",
            final_min_epochs=final_min_epochs,
            supervised_contrastive_weight=max(float(config.supervised_contrastive_weight), 0.02),
            supervised_contrastive_temperature=_min_positive_temperature(
                config.supervised_contrastive_temperature,
                0.20,
            ),
        )

    latent_head_refit = config.latent_head_refit
    latent_head_refit_selection_metric = config.latent_head_refit_selection_metric
    latent_head_refit_c_values = config.latent_head_refit_c_values
    latent_head_refit_blend_alphas = config.latent_head_refit_blend_alphas
    if preset in {"anti_collapse_head_refit", "anti_collapse_head_blend"}:
        # Keep the neural encoder/decoders as the representation learner, but
        # let a source-only, class-balanced logistic head handle the final
        # decision boundary.  The C value is selected on source-validation
        # participants only; no held-out-subject labels are used.
        latent_head_refit = (
            "validation_selected_source_logistic_blend"
            if preset == "anti_collapse_head_blend"
            else "validation_selected_source_logistic"
        )
        latent_head_refit_selection_metric = "balanced_top2_top3_rank_balance"
        latent_head_refit_c_values = (0.03, 0.1, 0.3, 1.0, 3.0)
        latent_head_refit_blend_alphas = (0.0, 0.25, 0.5, 0.75, 1.0)

    return replace(
        config,
        training_preset=preset,
        balanced_batch_sampling=True,
        subject_class_balanced_batch_sampling=True,
        label_smoothing=label_smoothing,
        input_dropout=input_dropout,
        prediction_balance_weight=prediction_balance_weight,
        prediction_balance_target_smoothing=1.0,
        prediction_balance_temperature=prediction_balance_temperature,
        logit_mean_center_weight=logit_mean_center_weight,
        class_bias_l2_weight=class_bias_l2_weight,
        soft_macro_recall_weight=soft_macro_recall_weight,
        validation_source_count=validation_source_count,
        validation_prediction_balance_weight=validation_prediction_balance_weight,
        validation_selection_metric="balanced_top2_top3_rank_balance",
        final_min_epochs=final_min_epochs,
        supervised_contrastive_temperature=_min_positive_temperature(config.supervised_contrastive_temperature, 0.20),
        score_calibration="validation_selected_guarded",
        score_calibration_selection_metric="balanced_top2_top3_rank_balance",
        score_calibration_guard_tolerance=min(float(config.score_calibration_guard_tolerance), 0.0),
        score_calibration_final_refit=True,
        prediction_postprocessing="validation_selected_balanced_assignment",
        prediction_postprocessing_selection_metric="balanced_top2_top3_rank_balance",
        prediction_postprocessing_guard_tolerance=min(
            float(config.prediction_postprocessing_guard_tolerance),
            0.0,
        ),
        latent_head_refit=latent_head_refit,
        latent_head_refit_selection_metric=latent_head_refit_selection_metric,
        latent_head_refit_c_values=latent_head_refit_c_values,
        latent_head_refit_blend_alphas=latent_head_refit_blend_alphas,
    )


def _resolve_device(device: str):
    torch, _nn, _F = _lazy_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_random_seeds(seed: int, *, num_threads: int) -> None:
    torch, _nn, _F = _lazy_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if num_threads > 0:
        torch.set_num_threads(int(num_threads))


class _LatentSubjectAutoencoderBase:
    pass


def _gradient_reverse(tensor, strength: float):
    """Return ``tensor`` unchanged in the forward pass and reverse its gradient.

    This is used for subject-adversarial latent regularization.  The subject
    classifier is trained normally, while the shared encoder receives the
    opposite gradient and is therefore discouraged from encoding source-subject
    identity.  A zero/negative strength is intentionally equivalent to no
    adversarial pressure.
    """

    torch, _nn, _F = _lazy_torch()

    class _GradientReverse(torch.autograd.Function):  # type: ignore[misc, name-defined]
        @staticmethod
        def forward(ctx, value, scale):
            ctx.scale = float(max(0.0, scale))
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.scale * grad_output, None

    return _GradientReverse.apply(tensor, float(strength))


def _make_model_class():
    torch, nn, _F = _lazy_torch()

    class LatentSubjectAutoencoder(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self, *, n_features: int, n_classes: int, subject_ids: Iterable[int], hidden_dim: int, latent_dim: int, dropout: float, input_dropout: float):
            super().__init__()
            self.input_dropout = nn.Dropout(float(input_dropout))
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.classifier = nn.Linear(latent_dim, n_classes)
            # The first latent-AE smoke run showed hard argmax collapse onto a
            # few classes.  A randomly initialized class bias can persist when
            # early stopping selects very early epochs, so start the classifier
            # from a neutral class prior and let source labels learn deviations.
            nn.init.xavier_uniform_(self.classifier.weight, gain=0.5)
            nn.init.zeros_(self.classifier.bias)
            subject_ids_tuple = tuple(int(subject_id) for subject_id in subject_ids)
            self.subject_classifier = nn.Linear(latent_dim, max(1, len(subject_ids_tuple)))
            max_subject_id = max(subject_ids_tuple) if subject_ids_tuple else 0
            subject_index_lookup = torch.full((max_subject_id + 1,), -1, dtype=torch.long)
            for subject_index, subject_id in enumerate(subject_ids_tuple):
                subject_index_lookup[int(subject_id)] = int(subject_index)
            self.register_buffer("_subject_index_lookup", subject_index_lookup, persistent=False)
            self.decoders = nn.ModuleDict(
                {
                    self._key(subject_id): nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, n_features),
                    )
                    for subject_id in subject_ids_tuple
                }
            )

        @staticmethod
        def _key(subject_id: int) -> str:
            return f"p{int(subject_id)}"

        def forward(self, features):
            latent = self.encoder(self.input_dropout(features))
            return self.classifier(latent), latent

        def reconstruct_subject(self, subject_id: int, latent):
            return self.decoders[self._key(int(subject_id))](latent)

        def subject_targets(self, participant_ids):
            participant_ids = participant_ids.to(device=self._subject_index_lookup.device, dtype=torch.long)
            if bool(torch.any(participant_ids < 0)) or bool(
                torch.any(participant_ids >= int(self._subject_index_lookup.shape[0]))
            ):
                raise ValueError("Encountered participant ids outside the subject-adversary lookup range.")
            targets = self._subject_index_lookup[participant_ids]
            if bool(torch.any(targets < 0)):
                raise ValueError("Encountered participant ids that were not part of the source-subject decoder set.")
            return targets

        def adversarial_subject_logits(self, latent, participant_ids, *, strength: float):
            reversed_latent = _gradient_reverse(latent, strength)
            return self.subject_classifier(reversed_latent), self.subject_targets(participant_ids)

    return LatentSubjectAutoencoder


def _concat_features(feature_sets, participants: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    participant_ids = []
    for participant in participants:
        feature_set = feature_sets[int(participant)]
        x = np.asarray(feature_set.features, dtype=np.float32)
        y = np.asarray(feature_set.labels, dtype=int)
        features.append(x)
        labels.append(y)
        participant_ids.append(np.full(y.shape[0], int(participant), dtype=int))
    return np.vstack(features), np.concatenate(labels), np.concatenate(participant_ids)


def _shuffle_labels_within_subjects(labels: np.ndarray, participants: np.ndarray, *, seed: int, context: Sequence[int]) -> np.ndarray:
    shuffled = np.asarray(labels, dtype=int).copy()
    for participant in sorted(set(int(value) for value in participants)):
        mask = participants == participant
        seed_values = [int(seed), *[int(value) for value in context], int(participant)]
        rng = np.random.default_rng(np.random.SeedSequence(seed_values))
        shuffled[mask] = rng.permutation(shuffled[mask])
    return shuffled


def _class_index(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(classes, labels)
    if np.any(positions < 0) or np.any(positions >= classes.shape[0]) or np.any(classes[positions] != labels):
        missing = sorted(set(int(value) for value in labels) - set(int(value) for value in classes))
        raise ValueError(f"Encountered labels not present in source training classes: {missing}")
    return positions.astype(np.int64)


def _class_weights(y_index: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_index, minlength=n_classes).astype(float)
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return weights / np.mean(weights)


def _bounded_label_smoothing(value: float) -> float:
    """Clamp label smoothing to PyTorch's valid cross-entropy range."""

    return min(max(float(value), 0.0), 0.999)


def _class_balanced_focal_cross_entropy(
    logits,
    targets,
    *,
    weight,
    label_smoothing: float,
    focal_gamma: float,
):
    """Return class-weighted CE with optional focal modulation."""

    torch, _nn, F = _lazy_torch()
    gamma = max(0.0, float(focal_gamma))
    smoothing = _bounded_label_smoothing(label_smoothing)
    if gamma <= 0.0:
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=smoothing,
        )
    per_example_loss = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        label_smoothing=smoothing,
        reduction="none",
    )
    probabilities = F.softmax(logits, dim=1)
    true_probabilities = probabilities.gather(1, targets.reshape(-1, 1)).reshape(-1)
    focal_weight = torch.pow(1.0 - torch.clamp(true_probabilities, min=1e-6, max=1.0), gamma)
    return (focal_weight * per_example_loss).mean()


def _class_margin_loss(logits, targets, *, margin: float):
    torch, _nn, F = _lazy_torch()
    if int(logits.shape[0]) == 0 or int(logits.shape[1]) <= 1:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    targets = targets.to(dtype=torch.long, device=logits.device)
    row_indices = torch.arange(int(logits.shape[0]), device=logits.device)
    true_logits = logits[row_indices, targets]
    negative_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=int(logits.shape[1])).to(dtype=torch.bool),
        -torch.inf,
    )
    best_negative_logits = torch.max(negative_logits, dim=1).values
    finite_mask = torch.isfinite(best_negative_logits)
    if not bool(torch.any(finite_mask)):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    losses = F.relu(float(margin) - (true_logits[finite_mask] - best_negative_logits[finite_mask]))
    return losses.mean()


def _prediction_balance_loss(logits, label_indices, *, target_smoothing: float, temperature: float = 1.0):
    """Penalize minibatch-level predicted-class collapse.

    The smoke fold showed hard-argmax collapse onto a few classes.  A normal
    temperature-1 softmax can still look fairly uniform when logit margins are
    small, so expose a balance temperature: values below 1 make this loss more
    sensitive to the hard-prediction distribution while remaining differentiable.
    """

    torch, _nn, F = _lazy_torch()
    if int(logits.shape[0]) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    temperature = max(float(temperature), 1e-6)
    probabilities = F.softmax(logits / temperature, dim=1)
    predicted_distribution = probabilities.mean(dim=0)
    label_distribution = (
        F.one_hot(label_indices, num_classes=int(logits.shape[1])).to(dtype=logits.dtype).mean(dim=0)
    )
    uniform_distribution = torch.full_like(predicted_distribution, 1.0 / float(logits.shape[1]))
    smoothing = min(max(float(target_smoothing), 0.0), 1.0)
    target_distribution = (1.0 - smoothing) * label_distribution + smoothing * uniform_distribution
    return F.mse_loss(predicted_distribution, target_distribution)


def _soft_macro_recall_loss(logits, label_indices):
    """Differentiable balanced-accuracy surrogate for minibatch training."""

    torch, _nn, F = _lazy_torch()
    if int(logits.shape[0]) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    probabilities = F.softmax(logits, dim=1)
    n_classes = int(logits.shape[1])
    one_hot = F.one_hot(label_indices, num_classes=n_classes).to(dtype=logits.dtype)
    class_counts = one_hot.sum(dim=0)
    represented = class_counts > 0
    if not bool(torch.any(represented)):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    soft_true_positive = (one_hot * probabilities).sum(dim=0)
    soft_recall = soft_true_positive[represented] / class_counts[represented].clamp_min(1.0)
    return 1.0 - soft_recall.mean()


def _soft_worst_class_recall_loss(logits, label_indices):
    """Differentiable pressure against zero-recall class collapse."""

    torch, _nn, F = _lazy_torch()
    if int(logits.shape[0]) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    label_indices = label_indices.to(dtype=torch.long, device=logits.device)
    probabilities = F.softmax(logits, dim=1)
    n_classes = int(logits.shape[1])
    one_hot = F.one_hot(label_indices, num_classes=n_classes).to(dtype=logits.dtype)
    class_counts = one_hot.sum(dim=0)
    represented = class_counts > 0
    if not bool(torch.any(represented)):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    soft_true_positive = (one_hot * probabilities).sum(dim=0)
    soft_recall = soft_true_positive[represented] / class_counts[represented].clamp_min(1.0)
    return 1.0 - torch.min(soft_recall)


def _logit_mean_center_loss(logits):
    """Penalize minibatch-level class-logit offsets before softmax.

    The existing prediction-balance loss acts on the average softmax
    distribution.  In the latent AE smoke run, the hard predictions collapsed
    onto a few classes even though many margins were modest; in that regime the
    mean softmax can look less imbalanced than the argmax histogram.  This loss
    directly discourages persistent class-specific logit offsets while preserving
    each trial's relative evidence pattern.
    """

    torch, _nn, _F = _lazy_torch()
    if int(logits.shape[0]) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    row_centered_logits = logits - logits.mean(dim=1, keepdim=True)
    mean_centered_logits = row_centered_logits.mean(dim=0)
    mean_centered_logits = mean_centered_logits - mean_centered_logits.mean()
    return torch.mean(mean_centered_logits.square())


def _confidence_penalty(logits):
    """Return a differentiable penalty for over-confident class posteriors."""

    torch, _nn, F = _lazy_torch()
    if int(logits.shape[0]) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    probabilities = F.softmax(logits, dim=1)
    entropy = -torch.sum(probabilities * torch.log(torch.clamp(probabilities, min=1e-8)), dim=1)
    max_entropy = torch.log(torch.tensor(float(logits.shape[1]), dtype=logits.dtype, device=logits.device))
    return torch.clamp(max_entropy - entropy.mean(), min=0.0)


def _supervised_contrastive_loss(latent, label_indices, *, temperature: float):
    """Supervised contrastive loss for class-preserving shared latent spaces.

    The source-only latent AE can reconstruct source-subject structure while still
    producing a class-biased classifier.  This optional loss directly encourages
    trials with the same stimulus label to occupy nearby latent locations across
    source subjects, while using different-label trials in the same minibatch as
    negatives.  It is source-only and uses no held-out-subject labels.
    """

    torch, _nn, F = _lazy_torch()
    if int(latent.shape[0]) <= 1:
        return torch.zeros((), dtype=latent.dtype, device=latent.device)
    labels = label_indices.reshape(-1, 1)
    positive_mask = labels.eq(labels.T)
    self_mask = torch.eye(int(latent.shape[0]), dtype=torch.bool, device=latent.device)
    positive_mask = positive_mask & ~self_mask
    if not bool(torch.any(positive_mask)):
        return torch.zeros((), dtype=latent.dtype, device=latent.device)

    normalized = F.normalize(latent, dim=1)
    scale = max(float(temperature), 1e-6)
    logits = normalized @ normalized.T / scale
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_counts = positive_mask.sum(dim=1)
    valid_rows = positive_counts > 0
    positive_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)
    mean_positive_log_prob = positive_log_prob[valid_rows] / positive_counts[valid_rows].to(dtype=latent.dtype)
    return -mean_positive_log_prob.mean()


def _balanced_epoch_indices(label_indices: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Return one epoch order that interleaves classes before batching.

    The latent autoencoder smoke run showed hard-prediction collapse onto a
    small subset of classes even though the source labels are globally balanced.
    Random pooled minibatches can still produce short-run class skew and noisy
    class-gradient estimates.  This helper keeps every training row exactly once
    per epoch, but constructs the epoch order by cycling through per-class
    shuffled buckets so most contiguous minibatches see all classes.
    """

    label_indices = np.asarray(label_indices, dtype=np.int64).ravel()
    if label_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    buckets: list[list[int]] = []
    for class_index in sorted(int(value) for value in np.unique(label_indices)):
        indices = np.flatnonzero(label_indices == class_index).astype(np.int64)
        indices = np.asarray(rng.permutation(indices), dtype=np.int64)
        buckets.append(indices.tolist())
    order: list[int] = []
    while any(buckets):
        cycle: list[int] = []
        for bucket in buckets:
            if bucket:
                cycle.append(bucket.pop())
        rng.shuffle(cycle)
        order.extend(cycle)
    return np.asarray(order, dtype=np.int64)


def _subject_class_balanced_epoch_indices(
    label_indices: np.ndarray,
    participant_ids: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an epoch order interleaving both source subjects and classes."""

    label_indices = np.asarray(label_indices, dtype=np.int64).ravel()
    participant_ids = np.asarray(participant_ids, dtype=np.int64).ravel()
    if label_indices.shape[0] != participant_ids.shape[0]:
        raise ValueError("label_indices and participant_ids must have the same length.")
    if label_indices.size == 0:
        return np.asarray([], dtype=np.int64)

    buckets: list[list[int]] = []
    for participant_id in sorted(int(value) for value in np.unique(participant_ids)):
        participant_mask = participant_ids == int(participant_id)
        for class_index in sorted(int(value) for value in np.unique(label_indices[participant_mask])):
            indices = np.flatnonzero(participant_mask & (label_indices == int(class_index))).astype(np.int64)
            if indices.size:
                indices = np.asarray(rng.permutation(indices), dtype=np.int64)
                buckets.append(indices.tolist())

    order: list[int] = []
    while any(buckets):
        cycle: list[int] = []
        for bucket in buckets:
            if bucket:
                cycle.append(bucket.pop())
        rng.shuffle(cycle)
        order.extend(cycle)
    return np.asarray(order, dtype=np.int64)


def _fit_pca(train_features: np.ndarray, test_features: np.ndarray | None, *, components_pca: int, seed: int):
    n_components = int(min(int(components_pca), train_features.shape[0], train_features.shape[1]))
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    train_latent_input = pca.fit_transform(train_features).astype(np.float32)
    test_latent_input = None if test_features is None else pca.transform(test_features).astype(np.float32)
    explained = float(100.0 * np.sum(pca.explained_variance_ratio_))
    return pca, train_latent_input, test_latent_input, n_components, explained


def _prediction_balance_penalty(predicted_labels: np.ndarray, classes: np.ndarray) -> float:
    """Return squared distance between predicted class frequencies and uniform."""

    predicted_indices = _class_index(np.asarray(predicted_labels, dtype=int), classes)
    counts = np.bincount(predicted_indices, minlength=int(classes.shape[0])).astype(float)
    frequencies = counts / max(1.0, float(np.sum(counts)))
    target = np.full(int(classes.shape[0]), 1.0 / float(classes.shape[0]), dtype=float)
    return float(np.sum((frequencies - target) ** 2))


def _split_source_participants(
    source_participants: Sequence[int],
    validation_source_count: int,
    *,
    strategy: str = "tail",
    seed: int = 0,
    anchor: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_participants = tuple(int(value) for value in source_participants)
    count = max(0, int(validation_source_count))
    if count == 0 or len(source_participants) <= count + 1:
        return source_participants, tuple()

    n_sources = len(source_participants)
    strategy = str(strategy or "tail").strip().lower().replace("-", "_")
    if strategy == "tail":
        validation_indices = tuple(range(n_sources - count, n_sources))
    elif strategy == "head":
        validation_indices = tuple(range(count))
    elif strategy == "spread":
        raw_indices = np.linspace(0, n_sources - 1, num=count + 2, dtype=int)[1:-1]
        validation_indices = tuple(dict.fromkeys(int(index) for index in raw_indices))
        candidate = 0
        while len(validation_indices) < count:
            validation_indices = (*validation_indices, candidate)
            validation_indices = tuple(dict.fromkeys(validation_indices))
            candidate += 1
    elif strategy == "rotating":
        start = 0 if anchor is None else abs(int(anchor)) % n_sources
        validation_indices = tuple((start + offset) % n_sources for offset in range(count))
    elif strategy == "round_robin":
        start = (int(seed) + (0 if anchor is None else int(anchor))) % n_sources
        validation_indices = tuple((start + offset) % n_sources for offset in range(count))
    elif strategy == "seeded_random":
        seed_values = [int(seed), int(count), int(n_sources)]
        if anchor is not None:
            seed_values.append(int(anchor))
        rng = np.random.default_rng(np.random.SeedSequence(seed_values))
        validation_indices = tuple(
            sorted(int(index) for index in rng.choice(n_sources, size=count, replace=False).tolist())
        )
    else:
        raise ValueError(
            "validation_source_strategy must be one of: tail, head, spread, rotating, round_robin, seeded_random"
        )

    validation = tuple(source_participants[index] for index in validation_indices[:count])
    train = tuple(participant for participant in source_participants if participant not in validation)
    return train, validation


def _final_refit_epochs(selected_epoch: int, config: LatentAutoencoderConfig) -> int:
    selected = max(1, int(selected_epoch))
    scaled = int(math.ceil(selected * max(0.0, float(config.final_epoch_multiplier))))
    floored = max(scaled, max(0, int(config.final_min_epochs)))
    return max(1, min(int(config.epochs), floored))


def _train_model(  # pylint: disable=too-many-arguments,too-many-locals
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_participants: np.ndarray,
    *,
    classes: np.ndarray,
    subject_ids: Sequence[int],
    config: LatentAutoencoderConfig,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    max_epochs: int | None = None,
):
    torch, _nn, F = _lazy_torch()
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    Model = _make_model_class()
    device = _resolve_device(config.device)
    max_epochs = int(max_epochs if max_epochs is not None else config.epochs)

    y_index = _class_index(train_labels, classes)
    weights = torch.tensor(_class_weights(y_index, len(classes)), dtype=torch.float32, device=device)
    x_tensor = torch.tensor(train_features, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_index, dtype=torch.long, device=device)
    p_tensor = torch.tensor(train_participants.astype(np.int64), dtype=torch.long, device=device)

    model = Model(
        n_features=train_features.shape[1],
        n_classes=len(classes),
        subject_ids=subject_ids,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        input_dropout=config.input_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_balanced = -math.inf
    best_validation_selection_score = -math.inf
    best_validation_prediction_balance_penalty = np.nan
    best_validation_metrics: dict[str, float] = {}
    epochs_since_improvement = 0
    history = []
    rng = np.random.default_rng(config.seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        if config.subject_class_balanced_batch_sampling:
            epoch_indices = _subject_class_balanced_epoch_indices(y_index, train_participants, rng=rng)
        elif config.balanced_batch_sampling:
            epoch_indices = _balanced_epoch_indices(y_index, rng=rng)
        else:
            epoch_indices = rng.permutation(train_features.shape[0])
        permutation = torch.tensor(epoch_indices, dtype=torch.long, device=device)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, int(permutation.shape[0]), int(config.batch_size)):
            batch_index = permutation[start : start + int(config.batch_size)]
            xb = x_tensor[batch_index]
            yb = y_tensor[batch_index]
            pb = p_tensor[batch_index]
            logits, latent = model(xb)
            class_loss = _class_balanced_focal_cross_entropy(
                logits,
                yb,
                weight=weights,
                label_smoothing=config.label_smoothing,
                focal_gamma=config.focal_loss_gamma,
            )
            reconstruction_losses = []
            for subject_id in torch.unique(pb).detach().cpu().numpy().tolist():
                mask = pb == int(subject_id)
                if bool(torch.any(mask)):
                    reconstruction = model.reconstruct_subject(int(subject_id), latent[mask])
                    reconstruction_losses.append(F.mse_loss(reconstruction, xb[mask]))
            reconstruction_loss = torch.stack(reconstruction_losses).mean() if reconstruction_losses else torch.zeros((), device=device)
            loss = class_loss + float(config.reconstruction_weight) * reconstruction_loss
            if float(config.soft_macro_recall_weight) > 0.0:
                loss = loss + float(config.soft_macro_recall_weight) * _soft_macro_recall_loss(logits, yb)
            if float(config.soft_worst_class_recall_weight) > 0.0:
                loss = loss + float(config.soft_worst_class_recall_weight) * _soft_worst_class_recall_loss(logits, yb)
            if float(config.margin_loss_weight) > 0.0:
                margin_loss = _class_margin_loss(
                    logits,
                    yb,
                    margin=config.margin_loss_value,
                )
                loss = loss + float(config.margin_loss_weight) * margin_loss
            if float(config.subject_adversary_weight) > 0.0:
                subject_logits, subject_targets = model.adversarial_subject_logits(
                    latent,
                    pb,
                    strength=1.0,
                )
                loss = loss + float(config.subject_adversary_weight) * F.cross_entropy(subject_logits, subject_targets)
            if float(config.prediction_balance_weight) > 0.0:
                balance_loss = _prediction_balance_loss(
                    logits,
                    yb,
                    target_smoothing=config.prediction_balance_target_smoothing,
                    temperature=config.prediction_balance_temperature,
                )
                loss = loss + float(config.prediction_balance_weight) * balance_loss
            if float(config.logit_mean_center_weight) > 0.0:
                loss = loss + float(config.logit_mean_center_weight) * _logit_mean_center_loss(logits)
            if float(config.class_bias_l2_weight) > 0.0 and model.classifier.bias is not None:
                # Source labels are balanced, so persistent class-bias offsets are more likely collapse than signal.
                loss = loss + float(config.class_bias_l2_weight) * torch.mean(model.classifier.bias.square())
            if float(config.confidence_penalty_weight) > 0.0:
                loss = loss + float(config.confidence_penalty_weight) * _confidence_penalty(logits)
            if float(config.supervised_contrastive_weight) > 0.0:
                contrastive_loss = _supervised_contrastive_loss(
                    latent,
                    yb,
                    temperature=config.supervised_contrastive_temperature,
                )
                loss = loss + float(config.supervised_contrastive_weight) * contrastive_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1

        validation_balanced = np.nan
        validation_prediction_balance_penalty = np.nan
        validation_selection_score = np.nan
        validation_metrics: dict[str, float] = {}
        if validation is not None:
            validation_features, validation_labels = validation
            validation_scores = _predict_scores(model, validation_features, device=device, batch_size=config.batch_size)
            validation_metrics = _validation_selection_metrics(
                validation_labels,
                validation_scores,
                classes,
                config.validation_selection_metric,
            )
            validation_pred = classes[np.argmax(validation_scores, axis=1)]
            validation_balanced = float(validation_metrics["balanced_accuracy"])
            validation_prediction_balance_penalty = _prediction_balance_penalty(validation_pred, classes)
            validation_selection_score = float(validation_metrics["selection_score"]) - float(
                config.validation_prediction_balance_weight
            ) * validation_prediction_balance_penalty
            if validation_selection_score > best_validation_selection_score + 1e-8:
                best_validation_balanced = validation_balanced
                best_validation_selection_score = validation_selection_score
                best_validation_prediction_balance_penalty = validation_prediction_balance_penalty
                best_validation_metrics = validation_metrics
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if config.patience > 0 and epochs_since_improvement >= config.patience:
                break
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(1, batches),
                "validation_balanced_accuracy": validation_balanced,
                "validation_top2_accuracy": validation_metrics.get("top2_accuracy", np.nan),
                "validation_top3_accuracy": validation_metrics.get("top3_accuracy", np.nan),
                "validation_mean_true_label_rank": validation_metrics.get("mean_true_label_rank", np.nan),
                "validation_prediction_balance_score": validation_metrics.get("prediction_balance_score", np.nan),
                "validation_prediction_balance_penalty": validation_prediction_balance_penalty,
                "validation_selection_score": validation_selection_score,
            }
        )

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_balanced_accuracy": float(best_validation_balanced),
        "best_validation_selection_score": float(best_validation_selection_score),
        "best_validation_prediction_balance_penalty": float(best_validation_prediction_balance_penalty),
        "best_validation_top2_accuracy": float(best_validation_metrics.get("top2_accuracy", np.nan)),
        "best_validation_top3_accuracy": float(best_validation_metrics.get("top3_accuracy", np.nan)),
        "best_validation_mean_true_label_rank": float(best_validation_metrics.get("mean_true_label_rank", np.nan)),
        "best_validation_prediction_balance_score": float(best_validation_metrics.get("prediction_balance_score", np.nan)),
        "history": history,
    }


def _predict_scores(model, features: np.ndarray, *, device, batch_size: int) -> np.ndarray:
    torch, _nn, _F = _lazy_torch()
    model.eval()
    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    scores = []
    with torch.no_grad():
        for start in range(0, int(x_tensor.shape[0]), int(batch_size)):
            logits, _latent = model(x_tensor[start : start + int(batch_size)])
            scores.append(logits.detach().cpu().numpy())
    return np.vstack(scores)


def _predict_latent(model, features: np.ndarray, *, device, batch_size: int) -> np.ndarray:
    """Return encoder latents for an already fitted latent-AE model."""

    torch, _nn, _F = _lazy_torch()
    model.eval()
    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    latents = []
    with torch.no_grad():
        for start in range(0, int(x_tensor.shape[0]), int(batch_size)):
            _logits, latent = model(x_tensor[start : start + int(batch_size)])
            latents.append(latent.detach().cpu().numpy())
    return np.vstack(latents)


def _empty_latent_head_refit_metadata(config: LatentAutoencoderConfig, status: str) -> dict:
    return {
        "latent_head_refit": config.latent_head_refit,
        "latent_head_refit_status": status,
        "latent_head_refit_selected_c": np.nan,
        "latent_head_refit_c_values": ";".join(str(float(value)) for value in config.latent_head_refit_c_values),
        "latent_head_refit_selected_blend_alpha": np.nan,
        "latent_head_refit_blend_alphas": ";".join(str(float(value)) for value in config.latent_head_refit_blend_alphas),
        "latent_head_refit_selection_metric": config.latent_head_refit_selection_metric,
        "latent_head_refit_validation_balanced_accuracy": np.nan,
        "latent_head_refit_validation_selection_score": np.nan,
    }


def _logistic_head_score_matrix(model: LogisticRegression, latent: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return ``model.decision_function`` columns aligned to ``classes``."""

    latent = np.asarray(latent, dtype=float)
    classes = np.asarray(classes, dtype=int)
    raw_scores = np.asarray(model.decision_function(latent), dtype=float)
    if raw_scores.ndim == 1:
        raw_scores = np.column_stack([-raw_scores, raw_scores])
    aligned = np.full((latent.shape[0], classes.shape[0]), -1e9, dtype=float)
    for source_index, class_label in enumerate(np.asarray(model.classes_, dtype=int)):
        matches = np.flatnonzero(classes == int(class_label))
        if matches.size:
            aligned[:, int(matches[0])] = raw_scores[:, source_index]
    return aligned


def _row_standardized_score_matrix(scores: np.ndarray) -> np.ndarray:
    """Return row-centered, row-scaled scores for safe neural/logistic blending."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (n_trials, n_classes).")
    centered = scores - np.mean(scores, axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return centered / scale


def _blend_score_matrices(base_scores: np.ndarray, refit_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Blend neural-head and logistic-probe scores using source-selected alpha."""

    base_scores = np.asarray(base_scores, dtype=float)
    refit_scores = np.asarray(refit_scores, dtype=float)
    if base_scores.shape != refit_scores.shape:
        raise ValueError("base_scores and refit_scores must have the same shape.")
    alpha = min(max(float(alpha), 0.0), 1.0)
    if alpha <= 0.0:
        return base_scores
    if alpha >= 1.0:
        return refit_scores
    return (1.0 - alpha) * _row_standardized_score_matrix(base_scores) + alpha * _row_standardized_score_matrix(refit_scores)


def _selected_latent_head_blend_alpha(metadata: dict, *, default: float = 1.0) -> float:
    value = metadata.get("latent_head_refit_selected_blend_alpha", default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value):
        value = float(default)
    return min(max(value, 0.0), 1.0)


def _apply_latent_head_refit_scores(base_scores: np.ndarray, refit_scores: np.ndarray, config: LatentAutoencoderConfig, metadata: dict) -> np.ndarray:
    if str(config.latent_head_refit or "none") == "validation_selected_source_logistic_blend":
        return _blend_score_matrices(base_scores, refit_scores, _selected_latent_head_blend_alpha(metadata))
    return np.asarray(refit_scores, dtype=float)


def _fit_latent_logistic_head(
    train_latent: np.ndarray,
    train_labels: np.ndarray,
    validation_latent: np.ndarray | None,
    validation_labels: np.ndarray | None,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
    *,
    selected_c: float | None = None,
    validation_base_scores: np.ndarray | None = None,
    selected_blend_alpha: float | None = None,
) -> tuple[LogisticRegression | None, dict]:
    """Fit a source-only multinomial logistic probe on frozen encoder latents."""

    method = str(config.latent_head_refit or "none")
    if method == "none":
        return None, _empty_latent_head_refit_metadata(config, "not_requested")
    if method not in LATENT_HEAD_REFIT_CHOICES:
        raise ValueError(f"latent_head_refit must be one of {LATENT_HEAD_REFIT_CHOICES}, got {method!r}")
    blend_method = method == "validation_selected_source_logistic_blend"

    c_values = tuple(float(value) for value in config.latent_head_refit_c_values if float(value) > 0.0)
    if not c_values:
        c_values = (1.0,)
    candidate_c_values: tuple[float, ...]
    if selected_c is not None and np.isfinite(float(selected_c)) and float(selected_c) > 0.0:
        candidate_c_values = (float(selected_c),)
    elif method == "source_logistic":
        candidate_c_values = (c_values[0],)
    else:
        candidate_c_values = c_values
    if selected_blend_alpha is not None and np.isfinite(float(selected_blend_alpha)):
        candidate_blend_alphas = (min(max(float(selected_blend_alpha), 0.0), 1.0),)
    elif blend_method:
        candidate_blend_alphas = tuple(
            min(max(float(alpha), 0.0), 1.0)
            for alpha in config.latent_head_refit_blend_alphas
        ) or (0.0, 0.5, 1.0)
    else:
        candidate_blend_alphas = (1.0,)

    best_model: LogisticRegression | None = None
    best_c = float(candidate_c_values[0])
    best_alpha = float(candidate_blend_alphas[0])
    best_balanced = -math.inf
    best_selection = -math.inf
    for c_value in candidate_c_values:
        model = LogisticRegression(C=float(c_value), class_weight="balanced", max_iter=1000, n_jobs=1)
        model.fit(np.asarray(train_latent, dtype=float), np.asarray(train_labels, dtype=int))
        if validation_latent is not None and validation_labels is not None and len(validation_labels):
            logistic_scores = _logistic_head_score_matrix(model, validation_latent, classes)
            score_candidates = []
            if blend_method and validation_base_scores is not None:
                for alpha in candidate_blend_alphas:
                    score_candidates.append((float(alpha), _blend_score_matrices(validation_base_scores, logistic_scores, alpha)))
            else:
                score_candidates.append((1.0, logistic_scores))
        else:
            score_candidates = [(best_alpha, None)]
        for alpha, candidate_scores in score_candidates:
            if candidate_scores is not None:
                metrics = _validation_selection_metrics(
                    np.asarray(validation_labels, dtype=int),
                    candidate_scores,
                    classes,
                    config.latent_head_refit_selection_metric,
                )
                balanced = float(metrics["balanced_accuracy"])
                selection = float(metrics["selection_score"])
            else:
                balanced = np.nan
                selection = -abs(math.log(float(c_value))) - 0.001 * abs(float(alpha) - 1.0)
            if best_model is None or selection > best_selection + 1e-12 or (
                abs(selection - best_selection) <= 1e-12 and (np.isnan(best_balanced) or balanced > best_balanced + 1e-12)
            ):
                best_model = model
                best_c = float(c_value)
                best_alpha = float(alpha)
                best_balanced = float(balanced)
                best_selection = float(selection)

    metadata = _empty_latent_head_refit_metadata(config, "ok")
    metadata.update(
        {
            "latent_head_refit_selected_c": best_c,
            "latent_head_refit_selected_blend_alpha": best_alpha,
            "latent_head_refit_validation_balanced_accuracy": best_balanced,
            "latent_head_refit_validation_selection_score": best_selection,
        }
    )
    return best_model, metadata


def _true_label_ranks(true_labels: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    ranks = np.full(true_labels.shape[0], np.nan, dtype=float)
    sorted_indices = np.argsort(-scores, axis=1)
    sorted_classes = classes[sorted_indices]
    for row, label in enumerate(true_labels):
        matches = np.flatnonzero(sorted_classes[row] == int(label))
        if matches.size:
            ranks[row] = float(matches[0] + 1)
    return ranks


def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denominator = np.sum(exp_scores, axis=1, keepdims=True)
    denominator[denominator <= 0.0] = 1.0
    return exp_scores / denominator


def _prediction_balance_score(predicted_labels: np.ndarray, classes: np.ndarray) -> float:
    """Return a 0..1 prediction-balance score, where 1 is perfectly uniform."""

    predicted_labels = np.asarray(predicted_labels, dtype=int)
    if predicted_labels.size == 0:
        return 0.0
    n_classes = int(classes.shape[0])
    counts = np.bincount(_class_index(predicted_labels, classes), minlength=n_classes).astype(float)
    probabilities = counts / max(1.0, float(np.sum(counts)))
    uniform = np.full(n_classes, 1.0 / float(n_classes), dtype=float)
    l1_balance = 1.0 - 0.5 * float(np.sum(np.abs(probabilities - uniform)))
    nonzero = probabilities[probabilities > 0.0]
    entropy_balance = 0.0
    if nonzero.size:
        entropy_balance = -float(np.sum(nonzero * np.log(nonzero))) / math.log(float(n_classes))
    return float(np.clip(0.5 * (l1_balance + entropy_balance), 0.0, 1.0))


def _rank_score_from_ranks(ranks: np.ndarray, *, n_classes: int) -> float:
    finite = np.asarray(ranks, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    mean_rank = float(np.mean(finite))
    return float(np.clip(1.0 - ((mean_rank - 1.0) / max(1.0, float(n_classes - 1))), 0.0, 1.0))


def _row_rank_logits(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (n_trials, n_classes).")
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty_like(scores, dtype=float)
    row_indices = np.arange(int(scores.shape[0]))[:, None]
    ranks[row_indices, order] = np.arange(int(scores.shape[1]), dtype=float)
    rank_logits = -ranks / max(float(temperature), 1e-6)
    return rank_logits - np.mean(rank_logits, axis=1, keepdims=True)


def _validation_selection_metrics(
    true_labels: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    metric: str,
) -> dict[str, float]:
    true_labels = np.asarray(true_labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = classes[np.argmax(scores, axis=1)]
    ranks = _true_label_ranks(true_labels, scores, classes)
    balanced = float(balanced_accuracy_score(true_labels, predictions))
    top2 = float(np.mean(ranks <= 2))
    top3 = float(np.mean(ranks <= 3))
    rank_score = _rank_score_from_ranks(ranks, n_classes=int(classes.shape[0]))
    balance_score = _prediction_balance_score(predictions, classes)
    metric = str(metric or "balanced_accuracy")
    if metric == "balanced_accuracy":
        selection_score = balanced
    elif metric == "balanced_top2_top3_rank":
        selection_score = balanced + 0.20 * top2 + 0.10 * top3 + 0.10 * rank_score
    elif metric == "balanced_top2_top3_rank_balance":
        selection_score = balanced + 0.20 * top2 + 0.10 * top3 + 0.10 * rank_score + 0.05 * balance_score
    else:
        raise ValueError(
            "validation_selection_metric must be one of: "
            "balanced_accuracy, balanced_top2_top3_rank, balanced_top2_top3_rank_balance"
        )
    return {
        "selection_score": float(selection_score),
        "balanced_accuracy": balanced,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "mean_true_label_rank": float(np.nanmean(ranks)),
        "rank_score": rank_score,
        "prediction_balance_score": balance_score,
    }


def _validation_selection_metrics_from_predictions(
    true_labels: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    predicted_labels: np.ndarray,
    metric: str,
) -> dict[str, float]:
    """Score source-validation predictions using rank-aware objectives."""

    true_labels = np.asarray(true_labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    ranks = _true_label_ranks(true_labels, scores, classes)
    balanced = float(balanced_accuracy_score(true_labels, predicted_labels))
    top2 = float(np.mean(ranks <= 2))
    top3 = float(np.mean(ranks <= 3))
    rank_score = _rank_score_from_ranks(ranks, n_classes=int(classes.shape[0]))
    balance_score = _prediction_balance_score(predicted_labels, classes)
    metric = str(metric or "balanced_accuracy")
    if metric == "balanced_accuracy":
        selection_score = balanced
    elif metric == "balanced_top2_top3_rank":
        selection_score = balanced + 0.20 * top2 + 0.10 * top3 + 0.10 * rank_score
    elif metric == "balanced_top2_top3_rank_balance":
        selection_score = (
            balanced + 0.20 * top2 + 0.10 * top3 + 0.10 * rank_score + 0.05 * balance_score
        )
    else:
        raise ValueError(
            "prediction_postprocessing_selection_metric must be one of: "
            "balanced_accuracy, balanced_top2_top3_rank, balanced_top2_top3_rank_balance"
        )
    return {
        "selection_score": float(selection_score),
        "balanced_accuracy": balanced,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "mean_true_label_rank": float(np.nanmean(ranks)),
        "rank_score": rank_score,
        "prediction_balance_score": balance_score,
    }


def _empty_score_calibration_metadata(config: LatentAutoencoderConfig, status: str) -> dict:
    return {
        "score_calibration": config.score_calibration,
        "score_calibration_status": status,
        "score_calibration_selected_method": "none",
        "score_calibration_selected_candidates": "",
        "score_calibration_prior_source": "",
        "score_calibration_predicted_prior_source": "none",
        "score_calibration_alpha": 0.0,
        "score_calibration_temperature": 1.0,
        "score_calibration_logistic_c": np.nan,
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
        "score_calibration_validation_balanced_accuracy": np.nan,
        "score_calibration_uncalibrated_validation_balanced_accuracy": np.nan,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": np.nan,
        "score_calibration_uncalibrated_validation_selection_score": np.nan,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": 0.0,
        "score_calibration_bias_max": 0.0,
        "score_calibration_bias_mean_abs": 0.0,
        "score_calibration_scale_min": 1.0,
        "score_calibration_scale_max": 1.0,
        "score_calibration_scale_mean": 1.0,
        "score_calibration_confusion_smoothing": float(config.score_calibration_confusion_smoothing),
        "score_calibration_confusion_map_trace": np.nan,
        "score_calibration_vector_steps": ";".join(str(float(step)) for step in config.score_calibration_vector_steps),
        "score_calibration_vector_rounds": int(config.score_calibration_vector_rounds),
        "score_calibration_vector_l2": float(config.score_calibration_vector_l2),
        "score_calibration_vector_updates": 0,
        "score_calibration_final_refit": bool(config.score_calibration_final_refit),
        "score_calibration_final_refit_status": "not_requested",
        "score_calibration_final_refit_method": "none",
        "score_calibration_final_refit_prior_source": "",
        "score_calibration_final_refit_predicted_prior_source": "none",
        "score_calibration_final_refit_alpha": np.nan,
        "score_calibration_final_refit_temperature": np.nan,
        "score_calibration_final_refit_logistic_c": np.nan,
        "score_calibration_final_refit_balanced_accuracy": np.nan,
        "score_calibration_final_refit_uncalibrated_balanced_accuracy": np.nan,
        "score_calibration_final_refit_bias_mean_abs": np.nan,
    }


def _fit_validation_selected_score_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[np.ndarray | dict, dict]:
    """Select one guarded source-validation calibration method.

    The latent AE smoke run showed class-collapse, but the best correction can
    vary by held-out subject.  This selector tries several existing
    source-validation-only calibrators, scores them on the source-validation
    split, and applies only the best one to the held-out subject.  The held-out
    subject's main-task labels are never used.
    """

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    candidates = DEFAULT_LATENT_SELECTED_SCORE_CALIBRATION_CANDIDATES
    candidate_list = ";".join(candidates)
    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))

    best_calibration: np.ndarray | dict = np.zeros(int(classes.shape[0]), dtype=float)
    best_method = "none"
    best_balanced = uncalibrated_balanced
    best_selection = uncalibrated_selection
    best_objective = uncalibrated_selection
    best_rank = 0
    best_metadata = _empty_score_calibration_metadata(config, "ok")
    best_metadata.update(
        {
            "score_calibration_selected_method": "none",
            "score_calibration_selected_candidates": candidate_list,
            "score_calibration_validation_balanced_accuracy": uncalibrated_balanced,
            "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
            "score_calibration_validation_selection_score": uncalibrated_selection,
            "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        }
    )

    for rank, method in enumerate(candidates):
        method_config = replace(config, score_calibration=method)
        calibration, metadata = _fit_validation_score_calibration(
            validation_scores,
            validation_labels,
            classes,
            method_config,
        )
        calibrated_scores = _apply_score_calibration(validation_scores, calibration)
        metrics = _validation_selection_metrics(
            validation_labels,
            calibrated_scores,
            classes,
            config.score_calibration_selection_metric,
        )
        balanced = float(metrics["balanced_accuracy"])
        if balanced + 1e-12 < guard_floor:
            continue
        selection_score = float(metrics["selection_score"])
        objective = selection_score
        if (
            objective > best_objective + 1e-12
            or (abs(objective - best_objective) <= 1e-12 and balanced > best_balanced + 1e-12)
            or (
                abs(objective - best_objective) <= 1e-12
                and abs(balanced - best_balanced) <= 1e-12
                and rank < best_rank
            )
        ):
            best_calibration = calibration
            best_method = method
            best_balanced = balanced
            best_selection = selection_score
            best_objective = objective
            best_rank = rank
            best_metadata = dict(metadata)

    best_metadata.update(
        {
            "score_calibration": config.score_calibration,
            "score_calibration_status": "ok",
            "score_calibration_selected_method": best_method,
            "score_calibration_selected_candidates": candidate_list,
            "score_calibration_validation_balanced_accuracy": best_balanced,
            "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
            "score_calibration_validation_selection_score": best_selection,
            "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
            "score_calibration_selection_metric": config.score_calibration_selection_metric,
            "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        }
    )
    return best_calibration, best_metadata


def _fit_validation_score_calibration(
    validation_scores: np.ndarray | None,
    validation_labels: np.ndarray | None,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[np.ndarray | dict, dict]:
    """Fit a source-validation-only logit bias to reduce class collapse.

    The latent AE smoke run showed strong prediction-frequency imbalance.  This
    calibrator estimates the mismatch between true validation class prior and a
    model-implied prior, then tunes a scalar bias strength on the source-validation
    split only.  The guarded variant can optimize a richer rank/balance
    source-validation objective while rejecting any alpha whose validation
    balanced accuracy drops below the configured guard tolerance.
    Held-out-subject labels are never used.
    """

    zero_bias = np.zeros(int(classes.shape[0]), dtype=float)
    if config.score_calibration == "none":
        return zero_bias, _empty_score_calibration_metadata(config, "not_requested")
    supported_calibrations = {
        "validation_class_bias",
        "validation_class_bias_guarded",
        "validation_prediction_bias",
        "validation_argmax_class_bias",
        "validation_argmax_class_bias_guarded",
        "validation_confusion_blend",
        "validation_logistic_stack",
        "validation_logistic_stack_guarded",
        "validation_vector_bias",
        "validation_vector_bias_guarded",
        "validation_class_zscore",
        "validation_class_zscore_guarded",
        "validation_temperature_class_bias_guarded",
        "validation_temperature_argmax_class_bias_guarded",
        "validation_rank_prior_bias_guarded",
        "validation_score_standardize",
        "validation_score_standardize_guarded",
        "validation_selected_guarded",
    }
    if config.score_calibration not in supported_calibrations:
        raise ValueError(f"Unsupported latent AE score calibration: {config.score_calibration!r}")
    if validation_scores is None or validation_labels is None or len(validation_labels) == 0:
        return zero_bias, _empty_score_calibration_metadata(config, "no_validation")
    if config.score_calibration in {"validation_logistic_stack", "validation_logistic_stack_guarded"}:
        return _fit_validation_logistic_stack_calibration(
            validation_scores,
            validation_labels,
            classes,
            config,
        )
    if config.score_calibration == "validation_confusion_blend":
        return _fit_validation_confusion_blend_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration in {"validation_vector_bias", "validation_vector_bias_guarded"}:
        return _fit_validation_vector_bias_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration in {"validation_class_zscore", "validation_class_zscore_guarded"}:
        return _fit_validation_class_zscore_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration in {
        "validation_temperature_class_bias_guarded",
        "validation_temperature_argmax_class_bias_guarded",
    }:
        return _fit_validation_temperature_bias_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration == "validation_rank_prior_bias_guarded":
        return _fit_validation_rank_prior_bias_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration in {
        "validation_score_standardize",
        "validation_score_standardize_guarded",
    }:
        return _fit_validation_score_standardization_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration == "validation_selected_guarded":
        return _fit_validation_selected_score_calibration(
            validation_scores,
            validation_labels,
            classes,
            config,
        )

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    guarded_calibration = config.score_calibration.endswith("_guarded")
    label_indices = _class_index(validation_labels, classes)
    n_classes = int(classes.shape[0])
    smoothing = max(0.0, float(config.score_calibration_smoothing))

    true_counts = np.bincount(label_indices, minlength=n_classes).astype(float)
    true_prior = (true_counts + smoothing) / (float(np.sum(true_counts)) + smoothing * n_classes)

    if config.score_calibration in {"validation_class_bias", "validation_class_bias_guarded"}:
        prior_source = "mean_softmax"
        predicted_counts = np.sum(_softmax_scores(validation_scores), axis=0)
    else:
        # Directly target hard-argmax prediction collapse.  This is useful when
        # average softmax probabilities look only mildly imbalanced but argmax
        # predictions collapse onto a small subset of classes.
        prior_source = "argmax_predictions"
        predicted_counts = np.bincount(np.argmax(validation_scores, axis=1), minlength=n_classes).astype(float)
    predicted_prior = (predicted_counts + smoothing) / (float(np.sum(predicted_counts)) + smoothing * n_classes)
    base_bias = np.log(true_prior) - np.log(predicted_prior)
    base_bias = base_bias - float(np.mean(base_bias))

    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    alphas = tuple(float(alpha) for alpha in config.score_calibration_alphas)
    if not alphas:
        alphas = (1.0,)
    best_alpha = 0.0
    best_balanced = -math.inf
    best_selection = -math.inf
    best_objective = -math.inf
    if guarded_calibration:
        best_balanced = uncalibrated_balanced
        best_selection = uncalibrated_selection
        best_objective = uncalibrated_selection
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))
    for alpha in alphas:
        calibrated_scores = validation_scores + float(alpha) * base_bias
        calibrated_metrics = _validation_selection_metrics(
            validation_labels,
            calibrated_scores,
            classes,
            config.score_calibration_selection_metric,
        )
        balanced = float(calibrated_metrics["balanced_accuracy"])
        selection_score = float(calibrated_metrics["selection_score"])
        if guarded_calibration and balanced + 1e-12 < guard_floor:
            continue
        objective = selection_score if guarded_calibration else balanced
        if (
            objective > best_objective + 1e-12
            or (abs(objective - best_objective) <= 1e-12 and balanced > best_balanced + 1e-12)
            or (
                abs(objective - best_objective) <= 1e-12
                and abs(balanced - best_balanced) <= 1e-12
                and abs(float(alpha)) < abs(best_alpha)
            )
        ):
            best_objective = objective
            best_balanced = balanced
            best_selection = selection_score
            best_alpha = float(alpha)
    bias = best_alpha * base_bias
    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": prior_source,
        "score_calibration_predicted_prior_source": (
            "softmax_mean" if prior_source == "mean_softmax" else "argmax"
        ),
        "score_calibration_alpha": best_alpha,
        "score_calibration_temperature": 1.0,
        "score_calibration_logistic_c": np.nan,
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
        "score_calibration_validation_balanced_accuracy": best_balanced,
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": best_selection,
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": float(np.min(bias)),
        "score_calibration_bias_max": float(np.max(bias)),
        "score_calibration_bias_mean_abs": float(np.mean(np.abs(bias))),
    }
    return bias, metadata


def _refit_score_calibration_on_source_train(
    calibration: np.ndarray | dict,
    metadata: dict,
    source_scores: np.ndarray,
    source_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[np.ndarray | dict, dict]:
    """Optionally refit the selected source-only calibration on final source scores."""

    metadata = dict(metadata)
    metadata["score_calibration_final_refit"] = bool(config.score_calibration_final_refit)
    if not bool(config.score_calibration_final_refit):
        metadata.setdefault("score_calibration_final_refit_status", "not_requested")
        metadata.setdefault("score_calibration_final_refit_method", "none")
        return calibration, metadata

    if str(config.score_calibration or "none") == "none":
        metadata.update(
            {
                "score_calibration_final_refit_status": "not_requested",
                "score_calibration_final_refit_method": "none",
            }
        )
        return calibration, metadata

    selected_method = str(metadata.get("score_calibration_selected_method", "") or "")
    refit_method = selected_method if selected_method and selected_method != "none" else str(config.score_calibration)
    if refit_method in {"", "none", "validation_selected_guarded"}:
        metadata.update(
            {
                "score_calibration_final_refit_status": "skipped_no_selected_method",
                "score_calibration_final_refit_method": "none",
            }
        )
        return calibration, metadata

    refit_config = replace(config, score_calibration=refit_method)
    try:
        refit_calibration, refit_metadata = _fit_validation_score_calibration(
            source_scores,
            source_labels,
            classes,
            refit_config,
        )
    except ValueError:
        metadata.update(
            {
                "score_calibration_final_refit_status": "unsupported_method",
                "score_calibration_final_refit_method": refit_method,
            }
        )
        return calibration, metadata

    refit_status = str(refit_metadata.get("score_calibration_status", "ok"))
    metadata.update(
        {
            "score_calibration_final_refit_status": refit_status,
            "score_calibration_final_refit_method": refit_method,
            "score_calibration_final_refit_prior_source": refit_metadata.get("score_calibration_prior_source", ""),
            "score_calibration_final_refit_predicted_prior_source": refit_metadata.get(
                "score_calibration_predicted_prior_source",
                "none",
            ),
            "score_calibration_final_refit_alpha": refit_metadata.get("score_calibration_alpha", np.nan),
            "score_calibration_final_refit_temperature": refit_metadata.get("score_calibration_temperature", np.nan),
            "score_calibration_final_refit_logistic_c": refit_metadata.get("score_calibration_logistic_c", np.nan),
            "score_calibration_final_refit_balanced_accuracy": refit_metadata.get(
                "score_calibration_validation_balanced_accuracy",
                np.nan,
            ),
            "score_calibration_final_refit_uncalibrated_balanced_accuracy": refit_metadata.get(
                "score_calibration_uncalibrated_validation_balanced_accuracy",
                np.nan,
            ),
            "score_calibration_final_refit_bias_mean_abs": refit_metadata.get("score_calibration_bias_mean_abs", np.nan),
        }
    )
    if refit_status != "ok":
        return calibration, metadata
    return refit_calibration, metadata


def _fit_validation_temperature_bias_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[dict, dict]:
    """Fit source-validation temperature plus class-prior logit bias."""

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    label_indices = _class_index(validation_labels, classes)
    n_classes = int(classes.shape[0])
    smoothing = max(0.0, float(config.score_calibration_smoothing))

    true_counts = np.bincount(label_indices, minlength=n_classes).astype(float)
    true_prior = (true_counts + smoothing) / (float(np.sum(true_counts)) + smoothing * n_classes)

    if config.score_calibration == "validation_temperature_class_bias_guarded":
        prior_source = "mean_softmax"
        predicted_counts = np.sum(_softmax_scores(validation_scores), axis=0)
    else:
        prior_source = "argmax_predictions"
        predicted_counts = np.bincount(np.argmax(validation_scores, axis=1), minlength=n_classes).astype(float)
    predicted_prior = (predicted_counts + smoothing) / (float(np.sum(predicted_counts)) + smoothing * n_classes)
    base_bias = np.log(true_prior) - np.log(predicted_prior)
    base_bias = base_bias - float(np.mean(base_bias))

    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))

    alphas = tuple(float(alpha) for alpha in config.score_calibration_alphas) or (1.0,)
    temperatures = tuple(float(value) for value in config.score_calibration_temperatures if float(value) > 0.0) or (1.0,)

    best_alpha = 0.0
    best_temperature = 1.0
    best_balanced = uncalibrated_balanced
    best_selection = uncalibrated_selection
    best_objective = uncalibrated_selection
    for temperature in temperatures:
        scaled_scores = validation_scores / max(float(temperature), 1e-6)
        for alpha in alphas:
            calibrated_scores = scaled_scores + float(alpha) * base_bias
            calibrated_metrics = _validation_selection_metrics(
                validation_labels,
                calibrated_scores,
                classes,
                config.score_calibration_selection_metric,
            )
            balanced = float(calibrated_metrics["balanced_accuracy"])
            if balanced + 1e-12 < guard_floor:
                continue
            selection_score = float(calibrated_metrics["selection_score"])
            objective = selection_score
            if (
                objective > best_objective + 1e-12
                or (abs(objective - best_objective) <= 1e-12 and balanced > best_balanced + 1e-12)
                or (
                    abs(objective - best_objective) <= 1e-12
                    and abs(balanced - best_balanced) <= 1e-12
                    and abs(float(alpha)) + abs(float(temperature) - 1.0)
                    < abs(best_alpha) + abs(best_temperature - 1.0)
                )
            ):
                best_objective = objective
                best_balanced = balanced
                best_selection = selection_score
                best_alpha = float(alpha)
                best_temperature = float(temperature)

    bias = best_alpha * base_bias
    calibrator = {"kind": "temperature_bias", "temperature": best_temperature, "bias": bias}
    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": prior_source,
        "score_calibration_predicted_prior_source": (
            "softmax_mean" if prior_source == "mean_softmax" else "argmax"
        ),
        "score_calibration_alpha": best_alpha,
        "score_calibration_temperature": best_temperature,
        "score_calibration_logistic_c": np.nan,
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
        "score_calibration_validation_balanced_accuracy": best_balanced,
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": best_selection,
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": float(np.min(bias)),
        "score_calibration_bias_max": float(np.max(bias)),
        "score_calibration_bias_mean_abs": float(np.mean(np.abs(bias))),
    }
    return calibrator, metadata


def _fit_validation_rank_prior_bias_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[dict, dict]:
    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    label_indices = _class_index(validation_labels, classes)
    n_classes = int(classes.shape[0])
    smoothing = max(0.0, float(config.score_calibration_smoothing))

    true_counts = np.bincount(label_indices, minlength=n_classes).astype(float)
    true_prior = (true_counts + smoothing) / (float(np.sum(true_counts)) + smoothing * n_classes)
    predicted_counts = np.bincount(np.argmax(validation_scores, axis=1), minlength=n_classes).astype(float)
    predicted_prior = (predicted_counts + smoothing) / (float(np.sum(predicted_counts)) + smoothing * n_classes)
    base_bias = np.log(true_prior) - np.log(predicted_prior)
    base_bias = base_bias - float(np.mean(base_bias))

    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))
    alphas = tuple(float(alpha) for alpha in config.score_calibration_alphas) or (1.0,)
    temperatures = tuple(float(value) for value in config.score_calibration_temperatures if float(value) > 0.0) or (1.0,)

    best_alpha = 0.0
    best_temperature = 1.0
    best_balanced = uncalibrated_balanced
    best_selection = uncalibrated_selection
    best_objective = uncalibrated_selection
    for temperature in temperatures:
        rank_scores = _row_rank_logits(validation_scores, temperature=temperature)
        for alpha in alphas:
            calibrated_scores = rank_scores + float(alpha) * base_bias
            calibrated_metrics = _validation_selection_metrics(
                validation_labels,
                calibrated_scores,
                classes,
                config.score_calibration_selection_metric,
            )
            balanced = float(calibrated_metrics["balanced_accuracy"])
            if balanced + 1e-12 < guard_floor:
                continue
            selection_score = float(calibrated_metrics["selection_score"])
            objective = selection_score
            if (
                objective > best_objective + 1e-12
                or (abs(objective - best_objective) <= 1e-12 and balanced > best_balanced + 1e-12)
                or (
                    abs(objective - best_objective) <= 1e-12
                    and abs(balanced - best_balanced) <= 1e-12
                    and abs(float(alpha)) + abs(float(temperature) - 1.0)
                    < abs(best_alpha) + abs(best_temperature - 1.0)
                )
            ):
                best_objective = objective
                best_balanced = balanced
                best_selection = selection_score
                best_alpha = float(alpha)
                best_temperature = float(temperature)

    bias = best_alpha * base_bias
    calibrator = {"kind": "rank_prior_bias", "temperature": best_temperature, "bias": bias}
    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": "validation_rank_prior_bias",
        "score_calibration_predicted_prior_source": "argmax_rank_prior",
        "score_calibration_alpha": best_alpha,
        "score_calibration_temperature": best_temperature,
        "score_calibration_logistic_c": np.nan,
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
        "score_calibration_validation_balanced_accuracy": best_balanced,
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": best_selection,
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": float(np.min(bias)),
        "score_calibration_bias_max": float(np.max(bias)),
        "score_calibration_bias_mean_abs": float(np.mean(np.abs(bias))),
    }
    return calibrator, metadata


def _fit_validation_vector_bias_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[np.ndarray, dict]:
    """Fit a guarded per-class logit-bias vector on source-validation data.

    The scalar prior calibrators above move along one precomputed prior-mismatch
    direction.  That is intentionally conservative, but it can be too weak when
    the latent AE collapses onto several different attractor classes.  This
    source-only calibrator greedily adjusts one class bias at a time and accepts
    an update only when the source-validation objective improves.  The guarded
    variant additionally rejects any update whose validation balanced accuracy
    drops below the uncalibrated score minus ``score_calibration_guard_tolerance``.
    """

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    n_classes = int(classes.shape[0])
    if validation_scores.ndim != 2 or int(validation_scores.shape[1]) != n_classes:
        raise ValueError("validation_scores must have shape (n_trials, n_classes).")

    guarded_calibration = config.score_calibration.endswith("_guarded")
    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    best_bias = np.zeros(n_classes, dtype=float)
    best_balanced = uncalibrated_balanced
    best_selection = uncalibrated_selection
    best_objective = uncalibrated_selection if guarded_calibration else uncalibrated_balanced
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))
    l2_weight = max(0.0, float(config.score_calibration_vector_l2))

    steps = tuple(abs(float(step)) for step in config.score_calibration_vector_steps if abs(float(step)) > 0.0)
    if not steps:
        steps = (0.25,)
    rounds = max(1, int(config.score_calibration_vector_rounds))
    updates = 0

    def penalized_objective(objective: float, bias: np.ndarray) -> float:
        return float(objective) - l2_weight * float(np.mean(np.square(bias)))

    for _round_index in range(rounds):
        improved_this_round = False
        for step in steps:
            current_penalized = penalized_objective(best_objective, best_bias)
            candidate_state: tuple[float, float, float, np.ndarray] | None = None
            for class_index in range(n_classes):
                for direction in (-1.0, 1.0):
                    trial_bias = best_bias.copy()
                    trial_bias[class_index] += direction * step
                    # Bias vectors are identifiable only up to a constant.  Keep
                    # them centered so metadata and L2 regularization are stable.
                    trial_bias -= float(np.mean(trial_bias))
                    trial_scores = validation_scores + trial_bias.reshape(1, -1)
                    trial_metrics = _validation_selection_metrics(
                        validation_labels,
                        trial_scores,
                        classes,
                        config.score_calibration_selection_metric,
                    )
                    trial_balanced = float(trial_metrics["balanced_accuracy"])
                    if guarded_calibration and trial_balanced + 1e-12 < guard_floor:
                        continue
                    trial_selection = float(trial_metrics["selection_score"])
                    trial_objective = trial_selection if guarded_calibration else trial_balanced
                    trial_penalized = penalized_objective(trial_objective, trial_bias)
                    if (
                        trial_penalized > current_penalized + 1e-12
                        or (
                            abs(trial_penalized - current_penalized) <= 1e-12
                            and trial_balanced > best_balanced + 1e-12
                        )
                        or (
                            abs(trial_penalized - current_penalized) <= 1e-12
                            and abs(trial_balanced - best_balanced) <= 1e-12
                            and np.mean(np.abs(trial_bias)) < np.mean(np.abs(best_bias)) - 1e-12
                        )
                    ):
                        current_penalized = trial_penalized
                        candidate_state = (trial_objective, trial_balanced, trial_selection, trial_bias)
            if candidate_state is not None:
                best_objective, best_balanced, best_selection, best_bias = candidate_state
                updates += 1
                improved_this_round = True
        if not improved_this_round:
            break

    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": "validation_vector_bias",
        "score_calibration_predicted_prior_source": "greedy_class_bias",
        "score_calibration_alpha": 1.0 if updates else 0.0,
        "score_calibration_validation_balanced_accuracy": best_balanced,
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": best_selection,
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": float(np.min(best_bias)),
        "score_calibration_bias_max": float(np.max(best_bias)),
        "score_calibration_bias_mean_abs": float(np.mean(np.abs(best_bias))),
        "score_calibration_confusion_smoothing": float(config.score_calibration_confusion_smoothing),
        "score_calibration_confusion_map_trace": np.nan,
        "score_calibration_vector_steps": ";".join(str(float(step)) for step in steps),
        "score_calibration_vector_rounds": rounds,
        "score_calibration_vector_l2": l2_weight,
        "score_calibration_vector_updates": updates,
    }
    return best_bias, metadata


def _fit_validation_confusion_blend_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[dict, dict]:
    """Fit a conservative source-validation confusion-map blend."""

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    n_classes = int(classes.shape[0])
    true_indices = _class_index(validation_labels, classes)
    probabilities = _softmax_scores(validation_scores)
    predicted_indices = np.argmax(validation_scores, axis=1)
    smoothing = max(0.0, float(config.score_calibration_confusion_smoothing))
    confusion_counts = np.eye(n_classes, dtype=float) * smoothing
    for predicted_index, true_index in zip(predicted_indices, true_indices, strict=True):
        confusion_counts[int(predicted_index), int(true_index)] += 1.0
    row_sums = np.sum(confusion_counts, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    confusion_map = confusion_counts / row_sums

    uncalibrated_predictions = classes[predicted_indices]
    uncalibrated_balanced = float(balanced_accuracy_score(validation_labels, uncalibrated_predictions))
    alphas = tuple(float(alpha) for alpha in config.score_calibration_alphas) or (1.0,)
    best_alpha = 0.0
    best_balanced = -math.inf
    for alpha in alphas:
        alpha = min(max(float(alpha), 0.0), 1.0)
        calibrated_probabilities = (1.0 - alpha) * probabilities + alpha * (probabilities @ confusion_map)
        calibrated_predictions = classes[np.argmax(calibrated_probabilities, axis=1)]
        balanced = float(balanced_accuracy_score(validation_labels, calibrated_predictions))
        if balanced > best_balanced + 1e-12 or (abs(balanced - best_balanced) <= 1e-12 and abs(alpha) < abs(best_alpha)):
            best_balanced = balanced
            best_alpha = alpha

    calibrator = {
        "kind": "confusion_blend",
        "alpha": float(best_alpha),
        "confusion_map": confusion_map,
    }
    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": "validation_confusion_map",
        "score_calibration_predicted_prior_source": "confusion_map",
        "score_calibration_alpha": float(best_alpha),
        "score_calibration_logistic_c": np.nan,
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
        "score_calibration_validation_balanced_accuracy": float(best_balanced),
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": float(best_balanced),
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_balanced,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": 0.0,
        "score_calibration_bias_max": 0.0,
        "score_calibration_bias_mean_abs": 0.0,
        "score_calibration_confusion_smoothing": smoothing,
        "score_calibration_confusion_map_trace": float(np.trace(confusion_map)),
    }
    return calibrator, metadata


def _logistic_stack_score_matrix(model: LogisticRegression, scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return class-aligned log-probability scores from a fitted logistic stack."""

    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=int)
    probabilities = np.full((int(scores.shape[0]), int(classes.shape[0])), 1e-12, dtype=float)
    stacked_probabilities = model.predict_proba(scores)
    for local_index, label in enumerate(model.classes_):
        matches = np.flatnonzero(classes == int(label))
        if matches.size:
            probabilities[:, int(matches[0])] = stacked_probabilities[:, int(local_index)]
    np.clip(probabilities, 1e-12, 1.0, out=probabilities)
    row_sums = np.sum(probabilities, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return np.log(probabilities / row_sums)


def _fit_validation_logistic_stack_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[dict | np.ndarray, dict]:
    """Fit a source-validation-only logistic stack over latent class scores."""

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    classes = np.asarray(classes, dtype=int)
    zero_bias = np.zeros(int(classes.shape[0]), dtype=float)
    present_classes = set(int(value) for value in np.unique(validation_labels))
    missing_classes = sorted(set(int(value) for value in classes) - present_classes)
    if missing_classes:
        return zero_bias, {
            **_empty_score_calibration_metadata(config, "missing_validation_classes"),
            "score_calibration_prior_source": "validation_logistic_stack",
            "score_calibration_predicted_prior_source": "logistic_stack",
        }

    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    guarded_calibration = config.score_calibration.endswith("_guarded")
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))
    c_values = tuple(float(value) for value in config.score_calibration_logistic_c_values if float(value) > 0.0)
    if not c_values:
        c_values = (1.0,)

    best_model: LogisticRegression | None = None
    best_c = np.nan
    best_balanced = -math.inf
    best_selection = -math.inf
    best_objective = -math.inf
    for c_value in c_values:
        model = LogisticRegression(C=float(c_value), max_iter=2000, solver="lbfgs", class_weight="balanced")
        model.fit(validation_scores, validation_labels)
        calibrated_scores = _logistic_stack_score_matrix(model, validation_scores, classes)
        calibrated_metrics = _validation_selection_metrics(
            validation_labels,
            calibrated_scores,
            classes,
            config.score_calibration_selection_metric,
        )
        balanced = float(calibrated_metrics["balanced_accuracy"])
        selection_score = float(calibrated_metrics["selection_score"])
        if guarded_calibration and balanced + 1e-12 < guard_floor:
            continue
        objective = selection_score if guarded_calibration else balanced
        if (
            objective > best_objective + 1e-12
            or (abs(objective - best_objective) <= 1e-12 and balanced > best_balanced + 1e-12)
            or (
                abs(objective - best_objective) <= 1e-12
                and abs(balanced - best_balanced) <= 1e-12
                and (np.isnan(best_c) or float(c_value) < float(best_c))
            )
        ):
            best_objective = objective
            best_balanced = balanced
            best_selection = selection_score
            best_c = float(c_value)
            best_model = model

    if best_model is None:
        return zero_bias, {
            **_empty_score_calibration_metadata(config, "guard_rejected" if guarded_calibration else "fit_failed"),
            "score_calibration_prior_source": "validation_logistic_stack",
            "score_calibration_predicted_prior_source": "logistic_stack",
            "score_calibration_validation_balanced_accuracy": uncalibrated_balanced,
            "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
            "score_calibration_validation_selection_score": uncalibrated_selection,
            "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        }

    calibrator = {"kind": "logistic_stack", "model": best_model, "classes": classes}
    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": "validation_logistic_stack",
        "score_calibration_predicted_prior_source": "logistic_stack",
        "score_calibration_alpha": 0.0,
        "score_calibration_logistic_c": float(best_c),
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in c_values),
        "score_calibration_validation_balanced_accuracy": float(best_balanced),
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": float(best_selection),
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": 0.0,
        "score_calibration_bias_max": 0.0,
        "score_calibration_bias_mean_abs": 0.0,
        "score_calibration_confusion_smoothing": float(config.score_calibration_confusion_smoothing),
        "score_calibration_confusion_map_trace": np.nan,
        "score_calibration_scale_min": np.nan,
        "score_calibration_scale_max": np.nan,
    }
    return calibrator, metadata


def _fit_validation_score_standardization_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[dict, dict]:
    """Fit source-validation-only classwise score standardization.

    The latent AE smoke fold showed a hard argmax collapse onto a small subset
    of classes.  Additive prior/bias calibration can fix offset problems, but it
    cannot fix class-specific logit scale problems.  This calibrator estimates
    per-class validation score mean and standard deviation and tunes a blend
    between raw logits and classwise standardized logits on source-validation
    subjects only.  Held-out-subject labels are never used.
    """

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    guarded_calibration = str(config.score_calibration).endswith("_guarded")
    n_classes = int(classes.shape[0])
    means = np.mean(validation_scores, axis=0)
    scales = np.std(validation_scores, axis=0)
    positive_scales = scales[scales > 1e-6]
    fallback_scale = float(np.median(positive_scales)) if positive_scales.size else 1.0
    scales = np.where(scales > 1e-6, scales, fallback_scale)
    scales = np.maximum(scales, 1e-6)

    def _calibrated_scores(scores: np.ndarray, alpha: float) -> np.ndarray:
        standardized = (np.asarray(scores, dtype=float) - means.reshape(1, -1)) / scales.reshape(1, -1)
        return (1.0 - float(alpha)) * np.asarray(scores, dtype=float) + float(alpha) * standardized

    uncalibrated_metrics = _validation_selection_metrics(
        validation_labels,
        validation_scores,
        classes,
        config.score_calibration_selection_metric,
    )
    uncalibrated_balanced = float(uncalibrated_metrics["balanced_accuracy"])
    uncalibrated_selection = float(uncalibrated_metrics["selection_score"])
    alphas = tuple(min(max(float(alpha), 0.0), 1.0) for alpha in config.score_calibration_alphas)
    if not alphas:
        alphas = (1.0,)
    best_alpha = 0.0
    best_balanced = uncalibrated_balanced if guarded_calibration else -math.inf
    best_selection = uncalibrated_selection if guarded_calibration else -math.inf
    best_objective = uncalibrated_selection if guarded_calibration else -math.inf
    guard_floor = uncalibrated_balanced - max(0.0, float(config.score_calibration_guard_tolerance))
    for alpha in alphas:
        calibrated_scores = _calibrated_scores(validation_scores, alpha)
        calibrated_metrics = _validation_selection_metrics(
            validation_labels,
            calibrated_scores,
            classes,
            config.score_calibration_selection_metric,
        )
        balanced = float(calibrated_metrics["balanced_accuracy"])
        selection_score = float(calibrated_metrics["selection_score"])
        if guarded_calibration and balanced + 1e-12 < guard_floor:
            continue
        objective = selection_score if guarded_calibration else balanced
        if (
            objective > best_objective + 1e-12
            or (abs(objective - best_objective) <= 1e-12 and balanced > best_balanced + 1e-12)
            or (
                abs(objective - best_objective) <= 1e-12
                and abs(balanced - best_balanced) <= 1e-12
                and abs(float(alpha)) < abs(best_alpha)
            )
        ):
            best_objective = objective
            best_balanced = balanced
            best_selection = selection_score
            best_alpha = float(alpha)

    calibrator = {
        "kind": "score_standardize",
        "alpha": float(best_alpha),
        "mean": means,
        "scale": scales,
    }
    metadata = {
        "score_calibration": config.score_calibration,
        "score_calibration_status": "ok",
        "score_calibration_prior_source": "validation_score_moments",
        "score_calibration_predicted_prior_source": "score_standardization",
        "score_calibration_alpha": float(best_alpha),
        "score_calibration_temperature": 1.0,
        "score_calibration_logistic_c": np.nan,
        "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
        "score_calibration_validation_balanced_accuracy": float(best_balanced),
        "score_calibration_uncalibrated_validation_balanced_accuracy": uncalibrated_balanced,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": float(best_selection),
        "score_calibration_uncalibrated_validation_selection_score": uncalibrated_selection,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": float(np.min(-means)),
        "score_calibration_bias_max": float(np.max(-means)),
        "score_calibration_bias_mean_abs": float(np.mean(np.abs(means))),
        "score_calibration_confusion_smoothing": float(config.score_calibration_confusion_smoothing),
        "score_calibration_confusion_map_trace": np.nan,
        "score_calibration_scale_min": float(np.min(scales[:n_classes])),
        "score_calibration_scale_max": float(np.max(scales[:n_classes])),
        "score_calibration_scale_mean": float(np.mean(scales[:n_classes])),
    }
    return calibrator, metadata


def _fit_validation_class_zscore_calibration(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> tuple[dict, dict]:
    calibrator, metadata = _fit_validation_score_standardization_calibration(
        validation_scores,
        validation_labels,
        classes,
        config,
    )
    calibrator = dict(calibrator)
    calibrator["kind"] = "class_zscore"
    metadata = dict(metadata)
    metadata["score_calibration_prior_source"] = "validation_score_distribution"
    metadata["score_calibration_predicted_prior_source"] = "class_zscore"
    return calibrator, metadata


def _apply_score_calibration(scores: np.ndarray, calibration: np.ndarray | dict | None) -> np.ndarray:
    if calibration is None or len(calibration) == 0:
        return np.asarray(scores, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if isinstance(calibration, dict):
        kind = calibration.get("kind")
        if kind == "logistic_stack":
            return _logistic_stack_score_matrix(calibration["model"], scores, np.asarray(calibration["classes"], dtype=int))
        if kind in {"score_standardize", "class_zscore"}:
            alpha = min(max(float(calibration.get("alpha", 0.0)), 0.0), 1.0)
            means = np.asarray(calibration["mean"], dtype=float).reshape(1, -1)
            scales = np.asarray(calibration["scale"], dtype=float).reshape(1, -1)
            scales = np.maximum(scales, 1e-6)
            standardized = (scores - means) / scales
            return (1.0 - alpha) * scores + alpha * standardized
        if kind == "temperature_bias":
            temperature = max(float(calibration.get("temperature", 1.0)), 1e-6)
            bias = np.asarray(calibration["bias"], dtype=float).reshape(1, -1)
            return scores / temperature + bias
        if kind == "rank_prior_bias":
            temperature = max(float(calibration.get("temperature", 1.0)), 1e-6)
            bias = np.asarray(calibration["bias"], dtype=float).reshape(1, -1)
            return _row_rank_logits(scores, temperature=temperature) + bias
        if kind != "confusion_blend":
            raise ValueError(f"Unknown score calibration kind: {kind!r}")
        alpha = min(max(float(calibration.get("alpha", 0.0)), 0.0), 1.0)
        probabilities = _softmax_scores(scores)
        confusion_map = np.asarray(calibration["confusion_map"], dtype=float)
        calibrated_probabilities = (1.0 - alpha) * probabilities + alpha * (probabilities @ confusion_map)
        calibrated_probabilities = np.clip(calibrated_probabilities, 1e-12, 1.0)
        return np.log(calibrated_probabilities)
    return scores + np.asarray(calibration, dtype=float).reshape(1, -1)


def _display_label_map(classes: np.ndarray) -> dict[int, int]:
    """Return CSV display labels without double-shifting already 1-based labels."""

    class_values = np.asarray(classes, dtype=int).ravel()
    if (
        class_values.size
        and int(np.min(class_values)) == 0
        and int(np.max(class_values)) == class_values.size - 1
    ):
        return {int(label): int(label) + 1 for label in class_values}
    return {int(label): int(label) for label in class_values}


def _prediction_rows(  # pylint: disable=too-many-arguments
    *,
    test_participant: int,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
    pca_components: int,
    pca_explained_variance_percent: float,
) -> list[dict]:
    window_start, window_stop = _centered_window(config.window_center, config.window_size)
    display_labels = _display_label_map(classes)
    rows = []
    for trial_index, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
        row = {
            "test_participant": int(test_participant),
            "trial": int(trial_index),
            "true_label": int(true_label),
            "predicted_label": int(predicted_label),
            "true_stimulus": display_labels.get(int(true_label), int(true_label)),
            "predicted_stimulus": display_labels.get(int(predicted_label), int(predicted_label)),
            "correct": bool(int(true_label) == int(predicted_label)),
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "window_start_s": window_start,
            "window_stop_s": window_stop,
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "classifier": "latent_autoencoder",
            "latent_training_preset": config.training_preset,
            "latent_head_refit": config.latent_head_refit,
            "latent_head_refit_c_values": ";".join(str(float(value)) for value in config.latent_head_refit_c_values),
            "components_pca": config.components_pca,
            "actual_components_pca": pca_components,
            "pca_explained_variance_percent": pca_explained_variance_percent,
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "seed": config.seed,
            "latent_score_ensemble_size": len(_effective_ensemble_seeds(config)),
            "latent_score_ensemble_seeds": _format_seed_sequence(_effective_ensemble_seeds(config)),
            "input_dropout": config.input_dropout,
            "reconstruction_weight": config.reconstruction_weight,
            "subject_adversary_weight": config.subject_adversary_weight,
            "prediction_balance_weight": config.prediction_balance_weight,
            "prediction_balance_target_smoothing": config.prediction_balance_target_smoothing,
            "prediction_balance_temperature": config.prediction_balance_temperature,
            "logit_mean_center_weight": config.logit_mean_center_weight,
            "class_bias_l2_weight": config.class_bias_l2_weight,
            "confidence_penalty_weight": config.confidence_penalty_weight,
            "label_smoothing": config.label_smoothing,
            "focal_loss_gamma": config.focal_loss_gamma,
            "margin_loss_weight": config.margin_loss_weight,
            "margin_loss_value": config.margin_loss_value,
            "soft_macro_recall_weight": config.soft_macro_recall_weight,
            "soft_worst_class_recall_weight": config.soft_worst_class_recall_weight,
            "supervised_contrastive_weight": config.supervised_contrastive_weight,
            "supervised_contrastive_temperature": config.supervised_contrastive_temperature,
            "balanced_batch_sampling": config.balanced_batch_sampling,
            "score_calibration": config.score_calibration,
            "prediction_postprocessing": config.prediction_postprocessing,
            "label_shuffle_control": config.label_shuffle_control,
            "label_shuffle_seed": config.label_shuffle_seed if config.label_shuffle_control else np.nan,
        }
        for class_index, class_label in enumerate(classes):
            row[f"score_{display_labels.get(int(class_label), int(class_label))}"] = float(
                scores[trial_index, class_index]
            )
        rows.append(row)
    return rows


def _outer_row(  # pylint: disable=too-many-arguments
    *,
    test_participant: int,
    train_participants: Sequence[int],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
    pca_components: int,
    pca_explained_variance_percent: float,
    fit_metadata: dict,
    inner_validation_participants: Sequence[int],
) -> dict:
    ranks = _true_label_ranks(true_labels, scores, classes)
    finite_ranks = ranks[np.isfinite(ranks)]
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    balanced = float(balanced_accuracy_score(true_labels, predicted_labels))
    chance = 1.0 / float(config.chance_classes)
    window_start, window_stop = _centered_window(config.window_center, config.window_size)
    return {
        "outer_fold": int(test_participant),
        "test_participant": int(test_participant),
        "train_participants": ";".join(str(int(value)) for value in train_participants),
        "inner_validation_participants": ";".join(str(int(value)) for value in inner_validation_participants),
        "n_train_participants": int(len(train_participants)),
        "window_center_s": config.window_center,
        "window_size_s": config.window_size,
        "window_start_s": window_start,
        "window_stop_s": window_stop,
        "baseline_window_start_s": config.baseline_window[0],
        "baseline_window_stop_s": config.baseline_window[1],
        "feature_mode": config.feature_mode,
        "normalization": config.normalization,
        "alignment": "none",
        "classifier": "latent_autoencoder",
        "latent_training_preset": config.training_preset,
        "accuracy": accuracy,
        "latent_head_refit": config.latent_head_refit,
        "latent_head_refit_status": fit_metadata.get("latent_head_refit_status", "not_requested"),
        "latent_head_refit_selected_c": fit_metadata.get("latent_head_refit_selected_c", np.nan),
        "latent_head_refit_selected_blend_alpha": fit_metadata.get("latent_head_refit_selected_blend_alpha", np.nan),
        "latent_head_refit_blend_alphas": ";".join(str(float(alpha)) for alpha in config.latent_head_refit_blend_alphas),
        "latent_head_refit_c_values": ";".join(str(float(value)) for value in config.latent_head_refit_c_values),
        "latent_head_refit_validation_balanced_accuracy": fit_metadata.get("latent_head_refit_validation_balanced_accuracy", np.nan),
        "percent": 100.0 * accuracy,
        "balanced_accuracy": balanced,
        "balanced_percent": 100.0 * balanced,
        "top2_accuracy": float(np.mean(ranks <= 2)),
        "top2_percent": float(100.0 * np.mean(ranks <= 2)),
        "top3_accuracy": float(np.mean(ranks <= 3)),
        "top3_percent": float(100.0 * np.mean(ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
        "median_true_label_rank": float(np.median(finite_ranks)) if finite_ranks.size else np.nan,
        "chance_accuracy": chance,
        "chance_percent": 100.0 * chance,
        "top2_chance_accuracy": min(2.0 * chance, 1.0),
        "top2_chance_percent": min(200.0 * chance, 100.0),
        "top3_chance_accuracy": min(3.0 * chance, 1.0),
        "top3_chance_percent": min(300.0 * chance, 100.0),
        "chance_mean_rank": 0.5 * (config.chance_classes + 1.0),
        "above_chance": bool(balanced > chance),
        "n_test_trials": int(true_labels.shape[0]),
        "n_train_classes": int(classes.shape[0]),
        "n_test_classes": int(np.unique(true_labels).shape[0]),
        "components_pca": config.components_pca,
        "actual_components_pca": pca_components,
        "pca_explained_variance_percent": pca_explained_variance_percent,
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "input_dropout": config.input_dropout,
        "seed": config.seed,
        "latent_score_ensemble_size": int(fit_metadata.get("latent_score_ensemble_size", len(_effective_ensemble_seeds(config)))),
        "latent_score_ensemble_seeds": fit_metadata.get("latent_score_ensemble_seeds", _format_seed_sequence(_effective_ensemble_seeds(config))),
        "latent_score_ensemble_final_epochs": fit_metadata.get("latent_score_ensemble_final_epochs", ""),
        "reconstruction_weight": config.reconstruction_weight,
        "subject_adversary_weight": config.subject_adversary_weight,
        "prediction_balance_weight": config.prediction_balance_weight,
        "prediction_balance_target_smoothing": config.prediction_balance_target_smoothing,
        "prediction_balance_temperature": config.prediction_balance_temperature,
        "logit_mean_center_weight": config.logit_mean_center_weight,
        "class_bias_l2_weight": config.class_bias_l2_weight,
        "confidence_penalty_weight": config.confidence_penalty_weight,
        "label_smoothing": config.label_smoothing,
        "focal_loss_gamma": config.focal_loss_gamma,
        "margin_loss_weight": config.margin_loss_weight,
        "margin_loss_value": config.margin_loss_value,
        "soft_macro_recall_weight": config.soft_macro_recall_weight,
        "soft_worst_class_recall_weight": config.soft_worst_class_recall_weight,
        "supervised_contrastive_weight": config.supervised_contrastive_weight,
        "supervised_contrastive_temperature": config.supervised_contrastive_temperature,
        "balanced_batch_sampling": config.balanced_batch_sampling,
        "validation_prediction_balance_weight": config.validation_prediction_balance_weight,
        "score_calibration": config.score_calibration,
        "score_calibration_status": fit_metadata.get("score_calibration_status", "unknown"),
        "score_calibration_selected_method": fit_metadata.get("score_calibration_selected_method", "none"),
        "score_calibration_prior_source": fit_metadata.get("score_calibration_prior_source", ""),
        "prediction_postprocessing_selected_method": fit_metadata.get(
            "prediction_postprocessing_selected_method", "none"
        ),
        "score_calibration_predicted_prior_source": fit_metadata.get("score_calibration_predicted_prior_source", "none"),
        "score_calibration_alpha": fit_metadata.get("score_calibration_alpha", np.nan),
        "score_calibration_temperature": fit_metadata.get("score_calibration_temperature", np.nan),
        "score_calibration_temperatures": ";".join(str(float(value)) for value in config.score_calibration_temperatures),
        "score_calibration_validation_balanced_accuracy": fit_metadata.get("score_calibration_validation_balanced_accuracy", np.nan),
        "score_calibration_uncalibrated_validation_balanced_accuracy": fit_metadata.get(
            "score_calibration_uncalibrated_validation_balanced_accuracy", np.nan
        ),
        "score_calibration_selection_metric": fit_metadata.get(
            "score_calibration_selection_metric", config.score_calibration_selection_metric
        ),
        "score_calibration_validation_selection_score": fit_metadata.get("score_calibration_validation_selection_score", np.nan),
        "score_calibration_uncalibrated_validation_selection_score": fit_metadata.get(
            "score_calibration_uncalibrated_validation_selection_score", np.nan
        ),
        "score_calibration_guard_tolerance": fit_metadata.get(
            "score_calibration_guard_tolerance", config.score_calibration_guard_tolerance
        ),
        "score_calibration_vector_steps": fit_metadata.get(
            "score_calibration_vector_steps", ";".join(str(float(step)) for step in config.score_calibration_vector_steps)
        ),
        "score_calibration_vector_rounds": fit_metadata.get(
            "score_calibration_vector_rounds", config.score_calibration_vector_rounds
        ),
        "score_calibration_vector_l2": fit_metadata.get("score_calibration_vector_l2", config.score_calibration_vector_l2),
        "score_calibration_vector_updates": fit_metadata.get("score_calibration_vector_updates", 0),
        "score_calibration_final_refit": config.score_calibration_final_refit,
        "score_calibration_final_refit_status": fit_metadata.get("score_calibration_final_refit_status", "not_requested"),
        "score_calibration_final_refit_method": fit_metadata.get("score_calibration_final_refit_method", "none"),
        "score_calibration_final_refit_prior_source": fit_metadata.get("score_calibration_final_refit_prior_source", ""),
        "score_calibration_final_refit_predicted_prior_source": fit_metadata.get(
            "score_calibration_final_refit_predicted_prior_source", "none"
        ),
        "score_calibration_final_refit_alpha": fit_metadata.get("score_calibration_final_refit_alpha", np.nan),
        "score_calibration_final_refit_balanced_accuracy": fit_metadata.get("score_calibration_final_refit_balanced_accuracy", np.nan),
        "score_calibration_final_refit_bias_mean_abs": fit_metadata.get("score_calibration_final_refit_bias_mean_abs", np.nan),
        "prediction_postprocessing": config.prediction_postprocessing,
        "prediction_postprocessing_status": fit_metadata.get("prediction_postprocessing_status", "not_requested"),
        "prediction_postprocessing_quota_source": fit_metadata.get("prediction_postprocessing_quota_source", "none"),
        "prediction_postprocessing_class_quota_counts": fit_metadata.get("prediction_postprocessing_class_quota_counts", ""),
        "prediction_postprocessing_quota_strength": fit_metadata.get("prediction_postprocessing_quota_strength", np.nan),
        "prediction_postprocessing_apply": fit_metadata.get("prediction_postprocessing_apply", False),
        "prediction_postprocessing_objective_delta": fit_metadata.get("prediction_postprocessing_objective_delta", np.nan),
        "prediction_postprocessing_validation_balanced_accuracy": fit_metadata.get(
            "prediction_postprocessing_validation_balanced_accuracy", np.nan
        ),
        "prediction_postprocessing_uncalibrated_validation_balanced_accuracy": fit_metadata.get(
            "prediction_postprocessing_uncalibrated_validation_balanced_accuracy", np.nan
        ),
        "prediction_postprocessing_validation_objective_delta": fit_metadata.get(
            "prediction_postprocessing_validation_objective_delta", np.nan
        ),
        "prediction_postprocessing_validation_selection_score": fit_metadata.get(
            "prediction_postprocessing_validation_selection_score", np.nan
        ),
        "prediction_postprocessing_uncalibrated_validation_selection_score": fit_metadata.get(
            "prediction_postprocessing_uncalibrated_validation_selection_score", np.nan
        ),
        "prediction_postprocessing_selection_metric": fit_metadata.get(
            "prediction_postprocessing_selection_metric",
            config.prediction_postprocessing_selection_metric,
        ),
        "prediction_postprocessing_margin_threshold": fit_metadata.get(
            "prediction_postprocessing_margin_threshold", np.nan
        ),
        "prediction_postprocessing_fixed_predictions": fit_metadata.get(
            "prediction_postprocessing_fixed_predictions", 0
        ),
        "prediction_postprocessing_guard_tolerance": fit_metadata.get("prediction_postprocessing_guard_tolerance", np.nan),
        "prediction_postprocessing_shrinkage_alpha": fit_metadata.get("prediction_postprocessing_shrinkage_alpha", np.nan),
        "prediction_postprocessing_shrinkage_alphas": fit_metadata.get(
            "prediction_postprocessing_shrinkage_alphas", ";".join(str(float(alpha)) for alpha in config.prediction_postprocessing_shrinkage_alphas)
        ),
        "score_calibration_bias_min": fit_metadata.get("score_calibration_bias_min", np.nan),
        "score_calibration_bias_max": fit_metadata.get("score_calibration_bias_max", np.nan),
        "score_calibration_bias_mean_abs": fit_metadata.get("score_calibration_bias_mean_abs", np.nan),
        "score_calibration_confusion_smoothing": fit_metadata.get("score_calibration_confusion_smoothing", np.nan),
        "score_calibration_confusion_map_trace": fit_metadata.get("score_calibration_confusion_map_trace", np.nan),
        "score_calibration_scale_min": fit_metadata.get("score_calibration_scale_min", np.nan),
        "score_calibration_scale_max": fit_metadata.get("score_calibration_scale_max", np.nan),
        "score_calibration_scale_mean": fit_metadata.get("score_calibration_scale_mean", np.nan),
        "epochs_requested": config.epochs,
        "best_epoch": fit_metadata.get("best_epoch", np.nan),
        "final_epochs": fit_metadata.get("final_epochs", np.nan),
        "validation_selection_metric": config.validation_selection_metric,
        "best_validation_balanced_accuracy": fit_metadata.get("best_validation_balanced_accuracy", np.nan),
        "best_validation_selection_score": fit_metadata.get("best_validation_selection_score", np.nan),
        "best_validation_top2_accuracy": fit_metadata.get("best_validation_top2_accuracy", np.nan),
        "best_validation_top3_accuracy": fit_metadata.get("best_validation_top3_accuracy", np.nan),
        "best_validation_mean_true_label_rank": fit_metadata.get("best_validation_mean_true_label_rank", np.nan),
        "best_validation_prediction_balance_score": fit_metadata.get("best_validation_prediction_balance_score", np.nan),
        "best_validation_prediction_balance_penalty": fit_metadata.get(
            "best_validation_prediction_balance_penalty", np.nan
        ),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "prediction_balance_penalty": _prediction_balance_penalty(predicted_labels, classes),
        "validation_source_count": config.validation_source_count,
        "validation_source_strategy": config.validation_source_strategy,
        "refit_all_sources": config.refit_all_sources,
        "final_epoch_multiplier": config.final_epoch_multiplier,
        "final_min_epochs": config.final_min_epochs,
        "label_shuffle_control": config.label_shuffle_control,
        "label_shuffle_seed": config.label_shuffle_seed if config.label_shuffle_control else np.nan,
    }


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _one_sided_exact_sign_p_value(differences: np.ndarray) -> float:
    nonzero = np.asarray(differences, dtype=float)
    nonzero = nonzero[nonzero != 0.0]
    wins = int(np.sum(nonzero > 0.0))
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def _group_summary(outer_rows: list[dict], config: LatentAutoencoderConfig) -> list[dict]:
    if not outer_rows:
        return []
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in outer_rows], dtype=float)
    raw = np.asarray([float(row["accuracy"]) for row in outer_rows], dtype=float)
    top2 = np.asarray([float(row["top2_accuracy"]) for row in outer_rows], dtype=float)
    top3 = np.asarray([float(row["top3_accuracy"]) for row in outer_rows], dtype=float)
    ranks = np.asarray([float(row["mean_true_label_rank"]) for row in outer_rows], dtype=float)
    chance = float(outer_rows[0]["chance_accuracy"])
    differences = balanced - chance
    return [
        {
            "n_outer_folds": len(outer_rows),
            "n_test_participants": len(outer_rows),
            "window_center_s": config.window_center,
            "window_size_s": config.window_size,
            "baseline_window_start_s": config.baseline_window[0],
            "baseline_window_stop_s": config.baseline_window[1],
            "latent_training_preset": config.training_preset,
            "feature_mode": config.feature_mode,
            "normalization": config.normalization,
            "alignment": "none",
            "classifier": "latent_autoencoder",
            "components_pca": config.components_pca,
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "seed": config.seed,
            "latent_score_ensemble_size": len(_effective_ensemble_seeds(config)),
            "latent_score_ensemble_seeds": _format_seed_sequence(_effective_ensemble_seeds(config)),
            "reconstruction_weight": config.reconstruction_weight,
            "subject_adversary_weight": config.subject_adversary_weight,
            "prediction_balance_weight": config.prediction_balance_weight,
            "prediction_balance_target_smoothing": config.prediction_balance_target_smoothing,
            "prediction_balance_temperature": config.prediction_balance_temperature,
            "logit_mean_center_weight": config.logit_mean_center_weight,
            "class_bias_l2_weight": config.class_bias_l2_weight,
            "confidence_penalty_weight": config.confidence_penalty_weight,
            "label_smoothing": config.label_smoothing,
            "focal_loss_gamma": config.focal_loss_gamma,
            "margin_loss_weight": config.margin_loss_weight,
            "margin_loss_value": config.margin_loss_value,
            "soft_macro_recall_weight": config.soft_macro_recall_weight,
            "soft_worst_class_recall_weight": config.soft_worst_class_recall_weight,
            "supervised_contrastive_weight": config.supervised_contrastive_weight,
            "supervised_contrastive_temperature": config.supervised_contrastive_temperature,
            "balanced_batch_sampling": config.balanced_batch_sampling,
            "subject_class_balanced_batch_sampling": config.subject_class_balanced_batch_sampling,
            "validation_prediction_balance_weight": config.validation_prediction_balance_weight,
            "dropout": config.dropout,
            "input_dropout": config.input_dropout,
            "latent_head_refit": config.latent_head_refit,
            "latent_head_refit_c_values": ";".join(str(float(value)) for value in config.latent_head_refit_c_values),
            "latent_head_refit_selection_metric": config.latent_head_refit_selection_metric,
            "latent_head_refit_blend_alphas": ";".join(str(float(alpha)) for alpha in config.latent_head_refit_blend_alphas),
            "latent_head_refit_selected_blend_alpha_counts": _format_counter(
                Counter(str(row.get("latent_head_refit_selected_blend_alpha", "")) for row in outer_rows)
            ),
            "latent_head_refit_status_counts": _format_counter(
                Counter(row.get("latent_head_refit_status", "not_requested") for row in outer_rows)
            ),
            "latent_head_refit_selected_c_counts": _format_counter(
                Counter(str(row.get("latent_head_refit_selected_c", "")) for row in outer_rows)
            ),
            "score_calibration": config.score_calibration,
            "score_calibration_alphas": ";".join(str(float(alpha)) for alpha in config.score_calibration_alphas),
            "score_calibration_logistic_c_values": ";".join(str(float(value)) for value in config.score_calibration_logistic_c_values),
            "score_calibration_smoothing": config.score_calibration_smoothing,
            "score_calibration_confusion_smoothing": config.score_calibration_confusion_smoothing,
            "score_calibration_selection_metric": config.score_calibration_selection_metric,
            "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
            "score_calibration_vector_steps": ";".join(str(float(step)) for step in config.score_calibration_vector_steps),
            "score_calibration_vector_rounds": config.score_calibration_vector_rounds,
            "score_calibration_vector_l2": config.score_calibration_vector_l2,
            "score_calibration_final_refit": config.score_calibration_final_refit,
            "score_calibration_final_refit_status_counts": _format_counter(
                Counter(row.get("score_calibration_final_refit_status", "not_requested") for row in outer_rows)
            ),
            "score_calibration_final_refit_method_counts": _format_counter(
                Counter(row.get("score_calibration_final_refit_method", "none") for row in outer_rows)
            ),
            "score_calibration_status_counts": _format_counter(Counter(row.get("score_calibration_status", "unknown") for row in outer_rows)),
            "score_calibration_prior_source_counts": _format_counter(Counter(row.get("score_calibration_prior_source", "") for row in outer_rows)),
            "score_calibration_predicted_prior_source_counts": _format_counter(
                Counter(row.get("score_calibration_predicted_prior_source", "none") for row in outer_rows)
            ),
            "prediction_postprocessing": config.prediction_postprocessing,
            "prediction_postprocessing_status_counts": _format_counter(
                Counter(row.get("prediction_postprocessing_status", "not_requested") for row in outer_rows)
            ),
            "prediction_postprocessing_selected_method_counts": _format_counter(
                Counter(row.get("prediction_postprocessing_selected_method", "none") for row in outer_rows)
            ),
            "prediction_postprocessing_selection_metric": config.prediction_postprocessing_selection_metric,
            "prediction_postprocessing_guard_tolerance": config.prediction_postprocessing_guard_tolerance,
            "prediction_postprocessing_margin_thresholds": ";".join(
                str(float(threshold)) for threshold in config.prediction_postprocessing_margin_thresholds
            ),
            "prediction_postprocessing_margin_threshold_mean": float(
                np.nanmean([float(row.get("prediction_postprocessing_margin_threshold", np.nan)) for row in outer_rows])
            ),
            "prediction_postprocessing_fixed_predictions_mean": float(
                np.nanmean([float(row.get("prediction_postprocessing_fixed_predictions", np.nan)) for row in outer_rows])
            ),
            "epochs_requested": config.epochs,
            "validation_selection_metric": config.validation_selection_metric,
            "validation_source_count": config.validation_source_count,
            "validation_source_strategy": config.validation_source_strategy,
            "refit_all_sources": config.refit_all_sources,
            "final_epoch_multiplier": config.final_epoch_multiplier,
            "final_min_epochs": config.final_min_epochs,
            "label_shuffle_control": config.label_shuffle_control,
            "label_shuffle_seed": config.label_shuffle_seed if config.label_shuffle_control else np.nan,
            "chance_accuracy": chance,
            "chance_percent": 100.0 * chance,
            "accuracy_mean": float(np.mean(raw)),
            "accuracy_median": float(np.median(raw)),
            "accuracy_sem": _sem(raw),
            "percent_mean": float(100.0 * np.mean(raw)),
            "top2_accuracy_mean": float(np.mean(top2)),
            "top2_percent_mean": float(100.0 * np.mean(top2)),
            "top2_percent_sem": float(100.0 * _sem(top2)),
            "top3_accuracy_mean": float(np.mean(top3)),
            "top3_percent_mean": float(100.0 * np.mean(top3)),
            "top3_percent_sem": float(100.0 * _sem(top3)),
            "mean_true_label_rank_mean": float(np.mean(ranks)),
            "mean_true_label_rank_sem": _sem(ranks),
            "chance_mean_rank": 0.5 * (config.chance_classes + 1.0),
            "balanced_accuracy_mean": float(np.mean(balanced)),
            "balanced_accuracy_median": float(np.median(balanced)),
            "balanced_accuracy_sem": _sem(balanced),
            "balanced_percent_mean": float(100.0 * np.mean(balanced)),
            "balanced_percent_median": float(100.0 * np.median(balanced)),
            "balanced_percent_sem": float(100.0 * _sem(balanced)),
            "participants_above_chance": int(np.sum(balanced > chance)),
            "participants_total": int(balanced.size),
            "participants_at_or_below_chance": int(np.sum(balanced <= chance)),
            "one_sided_exact_sign_p_value": _one_sided_exact_sign_p_value(differences),
            "best_epoch_mean": float(np.mean([float(row["best_epoch"]) for row in outer_rows])),
            "final_epochs_mean": float(np.nanmean([float(row.get("final_epochs", np.nan)) for row in outer_rows])),
            "prediction_balance_penalty_mean": float(
                np.mean([float(row["prediction_balance_penalty"]) for row in outer_rows])
            ),
            "actual_components_pca_counts": _format_counter(Counter(int(row["actual_components_pca"]) for row in outer_rows)),
            "score_calibration_alpha_mean": float(np.nanmean([float(row.get("score_calibration_alpha", np.nan)) for row in outer_rows])),
            "score_calibration_logistic_c_mean": float(np.nanmean([float(row.get("score_calibration_logistic_c", np.nan)) for row in outer_rows])),
            "score_calibration_validation_selection_score_mean": float(
                np.nanmean([float(row.get("score_calibration_validation_selection_score", np.nan)) for row in outer_rows])
            ),
            "score_calibration_uncalibrated_validation_selection_score_mean": float(
                np.nanmean([float(row.get("score_calibration_uncalibrated_validation_selection_score", np.nan)) for row in outer_rows])
            ),
            "score_calibration_bias_mean_abs_mean": float(np.nanmean([float(row.get("score_calibration_bias_mean_abs", np.nan)) for row in outer_rows])),
            "score_calibration_confusion_map_trace_mean": float(
                np.nanmean([float(row.get("score_calibration_confusion_map_trace", np.nan)) for row in outer_rows])
            ),
            "score_calibration_final_refit_alpha_mean": float(
                np.nanmean([float(row.get("score_calibration_final_refit_alpha", np.nan)) for row in outer_rows])
            ),
            "score_calibration_final_refit_bias_mean_abs_mean": float(
                np.nanmean([float(row.get("score_calibration_final_refit_bias_mean_abs", np.nan)) for row in outer_rows])
            ),
            "score_calibration_final_refit_balanced_accuracy_mean": float(
                np.nanmean([float(row.get("score_calibration_final_refit_balanced_accuracy", np.nan)) for row in outer_rows])
            ),
            "prediction_postprocessing_validation_balanced_accuracy_mean": float(
                np.nanmean([float(row.get("prediction_postprocessing_validation_balanced_accuracy", np.nan)) for row in outer_rows])
            ),
            "prediction_postprocessing_uncalibrated_validation_balanced_accuracy_mean": float(
                np.nanmean([float(row.get("prediction_postprocessing_uncalibrated_validation_balanced_accuracy", np.nan)) for row in outer_rows])
            ),
            "prediction_postprocessing_validation_objective_delta_mean": float(
                np.nanmean([float(row.get("prediction_postprocessing_validation_objective_delta", np.nan)) for row in outer_rows])
            ),
            "prediction_postprocessing_shrinkage_alpha_mean": float(
                np.nanmean([float(row.get("prediction_postprocessing_shrinkage_alpha", np.nan)) for row in outer_rows])
            ),
            "best_validation_selection_score_mean": float(
                np.nanmean([float(row.get("best_validation_selection_score", np.nan)) for row in outer_rows])
            ),
            "best_validation_prediction_balance_score_mean": float(
                np.nanmean([float(row.get("best_validation_prediction_balance_score", np.nan)) for row in outer_rows])
            ),
        }
    ]


def _format_counter(counter: Counter) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=lambda value: str(value)))


def _source_prior_class_quotas(source_labels: np.ndarray, classes: np.ndarray, n_test_trials: int) -> np.ndarray:
    """Estimate integer target-class quotas from source labels.

    BUSH-MEG main-task folds are class-balanced, and the latent smoke run showed
    severe hard-argmax class collapse.  This helper supports an optional
    source-only postprocessor that uses the source-label prior to assign a fixed
    number of held-out trials to each class.  Held-out labels are never used.
    """

    n_test_trials = int(n_test_trials)
    n_classes = int(classes.shape[0])
    if n_test_trials < 0:
        raise ValueError("n_test_trials must be non-negative.")
    if n_classes == 0:
        return np.asarray([], dtype=int)
    source_indices = _class_index(np.asarray(source_labels, dtype=int), classes)
    source_counts = np.bincount(source_indices, minlength=n_classes).astype(float)
    if float(np.sum(source_counts)) <= 0.0:
        return _uniform_class_quotas(n_classes, n_test_trials)
    expected = source_counts / float(np.sum(source_counts)) * float(n_test_trials)
    return _round_expected_class_quotas(expected, n_test_trials)


def _source_prior_expected_class_counts(source_labels: np.ndarray, classes: np.ndarray, n_test_trials: int) -> np.ndarray:
    """Return fractional expected counts from the source-label class prior."""

    n_test_trials = int(n_test_trials)
    n_classes = int(classes.shape[0])
    if n_test_trials < 0:
        raise ValueError("n_test_trials must be non-negative.")
    if n_classes == 0:
        return np.asarray([], dtype=float)
    source_indices = _class_index(np.asarray(source_labels, dtype=int), classes)
    source_counts = np.bincount(source_indices, minlength=n_classes).astype(float)
    if float(np.sum(source_counts)) <= 0.0:
        return np.full(n_classes, float(n_test_trials) / float(n_classes), dtype=float)
    return source_counts / float(np.sum(source_counts)) * float(n_test_trials)


def _shrunk_source_prior_class_quotas(
    source_labels: np.ndarray,
    predicted_labels: np.ndarray,
    classes: np.ndarray,
    n_test_trials: int,
    *,
    shrinkage_alpha: float,
) -> np.ndarray:
    """Blend observed argmax counts with the source prior before assignment."""

    n_test_trials = int(n_test_trials)
    classes = np.asarray(classes, dtype=int)
    n_classes = int(classes.shape[0])
    if n_classes == 0:
        return np.asarray([], dtype=int)
    source_expected = _source_prior_expected_class_counts(source_labels, classes, n_test_trials)
    predicted_indices = _class_index(np.asarray(predicted_labels, dtype=int), classes)
    predicted_counts = np.bincount(predicted_indices, minlength=n_classes).astype(float)
    if float(np.sum(predicted_counts)) <= 0.0:
        predicted_counts = source_expected.copy()
    else:
        predicted_counts *= float(n_test_trials) / float(np.sum(predicted_counts))
    alpha = min(max(float(shrinkage_alpha), 0.0), 1.0)
    expected = (1.0 - alpha) * predicted_counts + alpha * source_expected
    return _round_expected_class_quotas(expected, n_test_trials)


def _blended_source_prior_class_quotas(
    predicted_labels: np.ndarray,
    source_labels: np.ndarray,
    classes: np.ndarray,
    n_test_trials: int,
    *,
    quota_strength: float,
) -> np.ndarray:
    return _shrunk_source_prior_class_quotas(
        source_labels,
        predicted_labels,
        classes,
        n_test_trials,
        shrinkage_alpha=quota_strength,
    )


def _uniform_class_quotas(n_classes: int, n_test_trials: int) -> np.ndarray:
    n_classes = int(n_classes)
    n_test_trials = int(n_test_trials)
    if n_classes <= 0:
        return np.asarray([], dtype=int)
    quotas = np.full(n_classes, n_test_trials // n_classes, dtype=int)
    quotas[: n_test_trials - int(np.sum(quotas))] += 1
    return quotas


def _round_expected_class_quotas(expected: np.ndarray, n_test_trials: int) -> np.ndarray:
    expected = np.asarray(expected, dtype=float)
    quotas = np.floor(expected).astype(int)
    remainder = int(n_test_trials) - int(np.sum(quotas))
    if remainder > 0:
        fractions = expected - quotas
        for index in np.argsort(-fractions)[:remainder]:
            quotas[int(index)] += 1
    elif remainder < 0:
        fractions = expected - quotas
        for index in np.argsort(fractions)[: abs(remainder)]:
            if quotas[int(index)] > 0:
                quotas[int(index)] -= 1
    return quotas.astype(int)


def _balanced_assignment_predictions(scores: np.ndarray, classes: np.ndarray, quotas: np.ndarray) -> tuple[np.ndarray, float]:
    """Assign predictions by maximizing total score under per-class quotas."""

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy is normally installed through sklearn.
        raise RuntimeError("prediction_postprocessing=source_prior_balanced_assignment requires scipy.") from exc

    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=int)
    quotas = np.asarray(quotas, dtype=int)
    if int(np.sum(quotas)) != int(scores.shape[0]):
        raise ValueError("Class quotas must sum to the number of scored trials.")
    repeated_class_indices = np.repeat(np.arange(int(classes.shape[0])), quotas)
    cost = -scores[:, repeated_class_indices]
    row_indices, assignment_columns = linear_sum_assignment(cost)
    predicted_indices = np.empty(int(scores.shape[0]), dtype=int)
    predicted_indices[row_indices] = repeated_class_indices[assignment_columns]
    argmax_score = float(np.sum(np.max(scores, axis=1)))
    assignment_score = float(np.sum(scores[np.arange(scores.shape[0]), predicted_indices]))
    return classes[predicted_indices], assignment_score - argmax_score


def _top_score_margins(scores: np.ndarray) -> np.ndarray:
    """Return top-1 minus top-2 logit margins for each scored trial."""

    scores = np.asarray(scores, dtype=float)
    if int(scores.shape[1]) <= 1:
        return np.full(int(scores.shape[0]), np.inf, dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def _low_margin_balanced_assignment_predictions(
    scores: np.ndarray,
    classes: np.ndarray,
    quotas: np.ndarray,
    *,
    margin_threshold: float,
) -> tuple[np.ndarray, float, int]:
    """Assign only low-margin trials while respecting source-prior quotas.

    Full balanced assignment is useful when the latent AE collapses hard argmax
    predictions onto a few classes, but it can also overwrite confident evidence.
    This variant keeps argmax predictions whose top-1/top-2 score margin exceeds a
    source-validation-selected threshold, then solves the quota assignment problem
    only for the remaining ambiguous trials.  If a high-margin class is already
    over quota, the least confident fixed rows from that class are released until
    the quota problem is feasible.
    """

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy is normally installed through sklearn.
        raise RuntimeError("low-margin balanced assignment requires scipy.") from exc

    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=int)
    quotas = np.asarray(quotas, dtype=int).copy()
    if int(np.sum(quotas)) != int(scores.shape[0]):
        raise ValueError("Class quotas must sum to the number of scored trials.")
    argmax_indices = np.argmax(scores, axis=1)
    margins = _top_score_margins(scores)
    fixed_mask = margins >= float(margin_threshold)

    # Keeping all high-margin predictions can make the quota problem infeasible
    # for over-predicted classes.  Release the least-confident fixed rows in those
    # classes; this preserves the strongest margins while guaranteeing feasibility.
    for class_index in range(int(classes.shape[0])):
        fixed_rows = np.flatnonzero(fixed_mask & (argmax_indices == class_index))
        overflow = int(fixed_rows.shape[0]) - int(quotas[class_index])
        if overflow > 0:
            rows_to_release = fixed_rows[np.argsort(margins[fixed_rows])[:overflow]]
            fixed_mask[rows_to_release] = False

    remaining_quotas = quotas.copy()
    fixed_counts = np.bincount(argmax_indices[fixed_mask], minlength=int(classes.shape[0]))
    remaining_quotas -= fixed_counts
    if np.any(remaining_quotas < 0):
        raise ValueError("Low-margin balanced assignment produced negative remaining quotas.")

    predicted_indices = np.empty(int(scores.shape[0]), dtype=int)
    predicted_indices[fixed_mask] = argmax_indices[fixed_mask]
    remaining_rows = np.flatnonzero(~fixed_mask)
    repeated_class_indices = np.repeat(np.arange(int(classes.shape[0])), remaining_quotas)
    if int(repeated_class_indices.shape[0]) != int(remaining_rows.shape[0]):
        raise ValueError("Remaining quotas must match the number of non-fixed trials.")
    if remaining_rows.size:
        cost = -scores[np.ix_(remaining_rows, repeated_class_indices)]
        row_indices, assignment_columns = linear_sum_assignment(cost)
        predicted_indices[remaining_rows[row_indices]] = repeated_class_indices[assignment_columns]
    argmax_score = float(np.sum(np.max(scores, axis=1)))
    assignment_score = float(np.sum(scores[np.arange(scores.shape[0]), predicted_indices]))
    return classes[predicted_indices], assignment_score - argmax_score, int(np.sum(fixed_mask))


def _validation_balanced_assignment_candidates(
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    source_labels: np.ndarray,
    classes: np.ndarray,
    config: LatentAutoencoderConfig,
) -> list[dict]:
    """Return validation-scored balanced-assignment postprocessor candidates.

    The latent autoencoder can collapse hard predictions onto a small class
    subset.  Existing postprocessors expose several manual fixes, but choosing a
    postprocessor from the held-out subject would be invalid.  This helper makes
    the choice source-only: evaluate raw argmax, source-prior assignment, and
    shrunk source-prior assignment on source-validation subjects only.  The
    selected candidate can then be applied to the true held-out subject without
    using held-out labels.
    """

    validation_scores = np.asarray(validation_scores, dtype=float)
    validation_labels = np.asarray(validation_labels, dtype=int)
    classes = np.asarray(classes, dtype=int)
    validation_argmax = classes[np.argmax(validation_scores, axis=1)]
    rows: list[dict] = []

    def _append_row(
        *,
        selected_method: str,
        predictions: np.ndarray,
        objective_delta: float,
        shrinkage_alpha: float,
        quota_source: str,
        quotas: np.ndarray | None,
        margin_threshold: float = np.nan,
        fixed_predictions: int = 0,
    ) -> None:
        metrics = _validation_selection_metrics_from_predictions(
            validation_labels,
            validation_scores,
            classes,
            predictions,
            config.prediction_postprocessing_selection_metric,
        )
        rows.append(
            {
                "selected_method": selected_method,
                "validation_predictions": np.asarray(predictions, dtype=int),
                "validation_balanced_accuracy": float(metrics["balanced_accuracy"]),
                "validation_selection_score": float(metrics["selection_score"]),
                "validation_selection_metric": config.prediction_postprocessing_selection_metric,
                "validation_prediction_balance_score": float(metrics["prediction_balance_score"]),
                "validation_objective_delta": float(objective_delta),
                "shrinkage_alpha": float(shrinkage_alpha) if np.isfinite(shrinkage_alpha) else np.nan,
                "quota_source": quota_source,
                "margin_threshold": float(margin_threshold) if np.isfinite(margin_threshold) else np.nan,
                "fixed_predictions": int(fixed_predictions),
                "quotas": None if quotas is None else np.asarray(quotas, dtype=int),
            }
        )

    _append_row(
        selected_method="none",
        predictions=validation_argmax,
        objective_delta=0.0,
        shrinkage_alpha=np.nan,
        quota_source="none",
        quotas=None,
    )
    source_quotas = _source_prior_class_quotas(source_labels, classes, int(validation_scores.shape[0]))
    source_assigned, source_objective_delta = _balanced_assignment_predictions(
        validation_scores,
        classes,
        source_quotas,
    )
    _append_row(
        selected_method="source_prior_balanced_assignment",
        predictions=source_assigned,
        objective_delta=source_objective_delta,
        shrinkage_alpha=np.nan,
        quota_source="source_label_prior",
        quotas=source_quotas,
    )
    margin_thresholds = tuple(float(threshold) for threshold in config.prediction_postprocessing_margin_thresholds)
    if not margin_thresholds:
        margin_thresholds = (0.0, 0.25, 0.5, 0.75, 1.0)
    for margin_threshold in margin_thresholds:
        low_margin_assigned, low_margin_objective_delta, fixed_predictions = _low_margin_balanced_assignment_predictions(
            validation_scores,
            classes,
            source_quotas,
            margin_threshold=margin_threshold,
        )
        _append_row(
            selected_method="source_prior_low_margin_balanced_assignment",
            predictions=low_margin_assigned,
            objective_delta=low_margin_objective_delta,
            shrinkage_alpha=np.nan,
            quota_source="source_label_prior_low_margin",
            quotas=source_quotas,
            margin_threshold=margin_threshold,
            fixed_predictions=fixed_predictions,
        )
    candidate_alphas = tuple(float(alpha) for alpha in config.prediction_postprocessing_shrinkage_alphas)
    if not candidate_alphas:
        candidate_alphas = (1.0,)
    for candidate_alpha in candidate_alphas:
        shrunk_quotas = _shrunk_source_prior_class_quotas(
            source_labels,
            validation_argmax,
            classes,
            int(validation_scores.shape[0]),
            shrinkage_alpha=candidate_alpha,
        )
        shrunk_assigned, shrunk_objective_delta = _balanced_assignment_predictions(
            validation_scores,
            classes,
            shrunk_quotas,
        )
        _append_row(
            selected_method="shrunk_source_prior_balanced_assignment",
            predictions=shrunk_assigned,
            objective_delta=shrunk_objective_delta,
            shrinkage_alpha=candidate_alpha,
            quota_source="shrunk_source_label_prior",
            quotas=shrunk_quotas,
        )
        for margin_threshold in margin_thresholds:
            (
                shrunk_low_margin_assigned,
                shrunk_low_margin_objective_delta,
                fixed_predictions,
            ) = _low_margin_balanced_assignment_predictions(
                validation_scores,
                classes,
                shrunk_quotas,
                margin_threshold=margin_threshold,
            )
            _append_row(
                selected_method="shrunk_source_prior_low_margin_balanced_assignment",
                predictions=shrunk_low_margin_assigned,
                objective_delta=shrunk_low_margin_objective_delta,
                shrinkage_alpha=candidate_alpha,
                quota_source="shrunk_source_label_prior_low_margin",
                quotas=shrunk_quotas,
                margin_threshold=margin_threshold,
                fixed_predictions=fixed_predictions,
            )

    # Conservative tie-breaker: prefer raw argmax if validation performance is
    # indistinguishable, then prefer lower assignment cost.  This makes the mode
    # safe to keep in exploratory runs.
    method_priority = {
        "none": 0,
        "source_prior_low_margin_balanced_assignment": 1,
        "shrunk_source_prior_balanced_assignment": 1,
        "shrunk_source_prior_low_margin_balanced_assignment": 1,
        "source_prior_balanced_assignment": 2,
    }
    rows.sort(
        key=lambda row: (
            row["validation_selection_score"],
            row["validation_balanced_accuracy"],
            row["validation_objective_delta"],
            -method_priority[row["selected_method"]],
        ),
        reverse=True,
    )
    return rows


def _postprocess_predictions(
    scores: np.ndarray,
    classes: np.ndarray,
    source_labels: np.ndarray,
    config: LatentAutoencoderConfig,
    *,
    validation_scores: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    method = str(config.prediction_postprocessing or "none")
    scores = np.asarray(scores, dtype=float)
    classes = np.asarray(classes, dtype=int)
    source_labels = np.asarray(source_labels, dtype=int)
    argmax_labels = classes[np.argmax(scores, axis=1)]
    base_metadata = {
        "prediction_postprocessing_quota_source": "none",
        "prediction_postprocessing_selected_method": "none",
        "prediction_postprocessing_class_quota_counts": "",
        "prediction_postprocessing_objective_delta": 0.0,
        "prediction_postprocessing_validation_balanced_accuracy": np.nan,
        "prediction_postprocessing_uncalibrated_validation_balanced_accuracy": np.nan,
        "prediction_postprocessing_validation_objective_delta": np.nan,
        "prediction_postprocessing_validation_selection_score": np.nan,
        "prediction_postprocessing_uncalibrated_validation_selection_score": np.nan,
        "prediction_postprocessing_guard_tolerance": float(config.prediction_postprocessing_guard_tolerance),
        "prediction_postprocessing_quota_strength": float(config.prediction_postprocessing_quota_strength),
        "prediction_postprocessing_selection_metric": config.prediction_postprocessing_selection_metric,
        "prediction_postprocessing_apply": False,
        "prediction_postprocessing_shrinkage_alpha": np.nan,
        "prediction_postprocessing_shrinkage_alphas": ";".join(str(float(alpha)) for alpha in config.prediction_postprocessing_shrinkage_alphas),
        "prediction_postprocessing_margin_threshold": np.nan,
        "prediction_postprocessing_fixed_predictions": 0,
        "prediction_postprocessing_margin_thresholds": ";".join(str(float(threshold)) for threshold in config.prediction_postprocessing_margin_thresholds),
    }
    if method == "none":
        return argmax_labels, {
            **base_metadata,
            "prediction_postprocessing_status": "not_requested",
        }
    supported = {
        "source_prior_balanced_assignment",
        "source_prior_soft_balanced_assignment",
        "validation_guarded_source_prior_balanced_assignment",
        "validation_guarded_source_prior_soft_balanced_assignment",
        "validation_guarded_shrunk_source_prior_balanced_assignment",
        "validation_selected_balanced_assignment",
    }
    if method not in supported:
        raise ValueError(
            "prediction_postprocessing must be one of: none, source_prior_balanced_assignment, "
            "source_prior_soft_balanced_assignment, validation_guarded_source_prior_balanced_assignment, "
            "validation_guarded_source_prior_soft_balanced_assignment, "
            "validation_guarded_shrunk_source_prior_balanced_assignment, "
            "validation_selected_balanced_assignment"
        )

    soft_quota_methods = {
        "source_prior_soft_balanced_assignment",
        "validation_guarded_source_prior_soft_balanced_assignment",
    }
    guarded_methods = {
        "validation_guarded_source_prior_balanced_assignment",
        "validation_guarded_source_prior_soft_balanced_assignment",
        "validation_guarded_shrunk_source_prior_balanced_assignment",
    }
    validation_metadata = dict(base_metadata)
    selected_shrinkage_alpha = 1.0
    if method == "validation_selected_balanced_assignment":
        if validation_scores is None or validation_labels is None or len(validation_labels) == 0:
            return argmax_labels, {
                **validation_metadata,
                "prediction_postprocessing_status": "no_validation",
            }
        validation_scores = np.asarray(validation_scores, dtype=float)
        validation_labels = np.asarray(validation_labels, dtype=int)
        candidate_rows = _validation_balanced_assignment_candidates(
            validation_scores,
            validation_labels,
            source_labels,
            classes,
            config,
        )
        selected_row = candidate_rows[0]
        unprocessed_row = next(row for row in candidate_rows if row["selected_method"] == "none")
        validation_metadata.update(
            {
                "prediction_postprocessing_status": "ok",
                "prediction_postprocessing_selected_method": selected_row["selected_method"],
                "prediction_postprocessing_quota_source": selected_row["quota_source"],
                "prediction_postprocessing_validation_balanced_accuracy": selected_row[
                    "validation_balanced_accuracy"
                ],
                "prediction_postprocessing_uncalibrated_validation_balanced_accuracy": unprocessed_row[
                    "validation_balanced_accuracy"
                ],
                "prediction_postprocessing_validation_objective_delta": selected_row[
                    "validation_objective_delta"
                ],
                "prediction_postprocessing_validation_selection_score": selected_row[
                    "validation_selection_score"
                ],
                "prediction_postprocessing_uncalibrated_validation_selection_score": unprocessed_row[
                    "validation_selection_score"
                ],
                "prediction_postprocessing_shrinkage_alpha": selected_row["shrinkage_alpha"],
                "prediction_postprocessing_margin_threshold": selected_row["margin_threshold"],
                "prediction_postprocessing_fixed_predictions": selected_row["fixed_predictions"],
            }
        )
        if selected_row["selected_method"] == "none":
            return argmax_labels, validation_metadata
        if selected_row["selected_method"] == "shrunk_source_prior_balanced_assignment":
            selected_shrinkage_alpha = float(selected_row["shrinkage_alpha"])
            quotas = _shrunk_source_prior_class_quotas(
                source_labels,
                argmax_labels,
                classes,
                int(scores.shape[0]),
                shrinkage_alpha=selected_shrinkage_alpha,
            )
            quota_source = "shrunk_source_label_prior"
            predicted_labels, objective_delta = _balanced_assignment_predictions(scores, classes, quotas)
            fixed_predictions = 0
        elif selected_row["selected_method"] == "shrunk_source_prior_low_margin_balanced_assignment":
            selected_shrinkage_alpha = float(selected_row["shrinkage_alpha"])
            quotas = _shrunk_source_prior_class_quotas(
                source_labels,
                argmax_labels,
                classes,
                int(scores.shape[0]),
                shrinkage_alpha=selected_shrinkage_alpha,
            )
            quota_source = "shrunk_source_label_prior_low_margin"
            predicted_labels, objective_delta, fixed_predictions = _low_margin_balanced_assignment_predictions(
                scores,
                classes,
                quotas,
                margin_threshold=float(selected_row["margin_threshold"]),
            )
        elif selected_row["selected_method"] == "source_prior_low_margin_balanced_assignment":
            quotas = _source_prior_class_quotas(source_labels, classes, int(scores.shape[0]))
            quota_source = "source_label_prior_low_margin"
            predicted_labels, objective_delta, fixed_predictions = _low_margin_balanced_assignment_predictions(
                scores,
                classes,
                quotas,
                margin_threshold=float(selected_row["margin_threshold"]),
            )
        else:
            quotas = _source_prior_class_quotas(source_labels, classes, int(scores.shape[0]))
            quota_source = "source_label_prior"
            predicted_labels, objective_delta = _balanced_assignment_predictions(scores, classes, quotas)
            fixed_predictions = 0
        quota_counts = Counter({int(class_label): int(quota) for class_label, quota in zip(classes, quotas, strict=True)})
        return predicted_labels, {
            **validation_metadata,
            "prediction_postprocessing_quota_source": quota_source,
            "prediction_postprocessing_class_quota_counts": _format_counter(quota_counts),
            "prediction_postprocessing_objective_delta": float(objective_delta),
            "prediction_postprocessing_fixed_predictions": int(fixed_predictions),
        }

    if method in guarded_methods:
        if validation_scores is None or validation_labels is None or len(validation_labels) == 0:
            return argmax_labels, {
                **validation_metadata,
                "prediction_postprocessing_status": "no_validation",
            }
        validation_scores = np.asarray(validation_scores, dtype=float)
        validation_labels = np.asarray(validation_labels, dtype=int)
        validation_argmax = classes[np.argmax(validation_scores, axis=1)]
        if method == "validation_guarded_shrunk_source_prior_balanced_assignment":
            candidate_alphas = tuple(float(alpha) for alpha in config.prediction_postprocessing_shrinkage_alphas)
            if not candidate_alphas:
                candidate_alphas = (1.0,)
            shrinkage_candidate_rows: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
            for candidate_alpha in candidate_alphas:
                validation_quotas = _shrunk_source_prior_class_quotas(
                    source_labels,
                    validation_argmax,
                    classes,
                    int(validation_scores.shape[0]),
                    shrinkage_alpha=candidate_alpha,
                )
                validation_assigned, validation_objective_delta = _balanced_assignment_predictions(
                    validation_scores,
                    classes,
                    validation_quotas,
                )
                validation_balanced = float(balanced_accuracy_score(validation_labels, validation_assigned))
                shrinkage_candidate_rows.append(
                    (
                        validation_balanced,
                        float(validation_objective_delta),
                        float(candidate_alpha),
                        validation_assigned,
                        validation_quotas,
                    )
                )
            shrinkage_candidate_rows.sort(key=lambda row: (row[0], row[1], -abs(row[2])), reverse=True)
            validation_balanced, validation_objective_delta, selected_shrinkage_alpha, validation_assigned, validation_quotas = shrinkage_candidate_rows[0]
            validation_metadata["prediction_postprocessing_shrinkage_alpha"] = float(selected_shrinkage_alpha)
        else:
            if method in soft_quota_methods:
                validation_quotas = _blended_source_prior_class_quotas(
                    validation_argmax,
                    source_labels,
                    classes,
                    int(validation_scores.shape[0]),
                    quota_strength=config.prediction_postprocessing_quota_strength,
                )
            else:
                validation_quotas = _source_prior_class_quotas(source_labels, classes, int(validation_scores.shape[0]))
            validation_assigned, validation_objective_delta = _balanced_assignment_predictions(
                validation_scores,
                classes,
                validation_quotas,
            )
            validation_balanced = float(balanced_accuracy_score(validation_labels, validation_assigned))
        uncalibrated_validation_balanced = float(balanced_accuracy_score(validation_labels, validation_argmax))
        validation_metadata.update(
            {
                "prediction_postprocessing_validation_balanced_accuracy": validation_balanced,
                "prediction_postprocessing_uncalibrated_validation_balanced_accuracy": uncalibrated_validation_balanced,
                "prediction_postprocessing_validation_objective_delta": float(validation_objective_delta),
            }
        )
        guard_tolerance = max(0.0, float(config.prediction_postprocessing_guard_tolerance))
        if validation_balanced + guard_tolerance + 1e-12 < uncalibrated_validation_balanced:
            return argmax_labels, {
                **validation_metadata,
                "prediction_postprocessing_status": "guard_rejected",
                "prediction_postprocessing_selected_method": "none",
            }

    if method in soft_quota_methods:
        quotas = _blended_source_prior_class_quotas(
            argmax_labels,
            source_labels,
            classes,
            int(scores.shape[0]),
            quota_strength=config.prediction_postprocessing_quota_strength,
        )
        quota_source = "argmax_source_prior_blend"
    elif method == "validation_guarded_shrunk_source_prior_balanced_assignment":
        quotas = _shrunk_source_prior_class_quotas(
            source_labels,
            argmax_labels,
            classes,
            int(scores.shape[0]),
            shrinkage_alpha=selected_shrinkage_alpha,
        )
        quota_source = "shrunk_source_label_prior"
    else:
        quotas = _source_prior_class_quotas(source_labels, classes, int(scores.shape[0]))
        quota_source = "source_label_prior"
    predicted_labels, objective_delta = _balanced_assignment_predictions(scores, classes, quotas)
    quota_counts = Counter({int(class_label): int(quota) for class_label, quota in zip(classes, quotas, strict=True)})
    return predicted_labels, {
        **validation_metadata,
        "prediction_postprocessing_status": "ok",
        "prediction_postprocessing_selected_method": method,
        "prediction_postprocessing_quota_source": quota_source,
        "prediction_postprocessing_class_quota_counts": _format_counter(quota_counts),
        "prediction_postprocessing_objective_delta": float(objective_delta),
        "prediction_postprocessing_apply": True,
    }


def evaluate_latent_autoencoder_loso(  # pylint: disable=too-many-locals
    data_folder,
    participants: Sequence[int],
    *,
    outer_participants: Sequence[int] | None = None,
    config: LatentAutoencoderConfig | None = None,
    progress=None,
) -> dict[str, list[dict]]:
    """Run strict source-only LOSO latent autoencoder evaluation."""

    config = config or LatentAutoencoderConfig()
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(value) for value in participants)
    outer_participants = tuple(int(value) for value in (outer_participants if outer_participants is not None else participants))
    unknown_outer = sorted(set(outer_participants) - set(participants))
    if unknown_outer:
        raise ValueError(f"Outer participants must be included in participants: {unknown_outer}")
    if len(participants) < 3:
        raise ValueError("At least three participants are required for LOSO decoding.")

    _set_random_seeds(config.seed, num_threads=config.num_threads)
    stimulus_config = CrossSubjectStimulusConfig(
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
    feature_sets = {}
    for participant in participants:
        if progress is not None:
            progress(f"LOAD participant={participant}")
        feature_sets[int(participant)] = load_participant_stimulus_features(data_folder, participant, config=stimulus_config)

    outer_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for outer_index, test_participant in enumerate(outer_participants, start=1):
        if progress is not None:
            progress(f"START outer_test_participant={test_participant} outer_index={outer_index}/{len(outer_participants)}")
        source_participants = tuple(participant for participant in participants if participant != test_participant)
        train_epoch_participants, validation_participants = _split_source_participants(
            source_participants,
            config.validation_source_count,
            strategy=config.validation_source_strategy,
            seed=config.seed,
            anchor=test_participant,
        )
        train_features_raw, train_labels_raw, train_subjects = _concat_features(feature_sets, train_epoch_participants)
        validation_tuple = None
        selected_epoch = config.epochs
        fit_metadata: dict = {"best_epoch": config.epochs, "best_validation_balanced_accuracy": np.nan}
        score_calibration_bias: np.ndarray | dict = np.zeros(0, dtype=float)
        initial_calibration_status = "not_requested" if config.score_calibration == "none" else "no_validation"
        score_calibration_metadata = _empty_score_calibration_metadata(config, initial_calibration_status)
        validation_scores_for_postprocessing = None
        validation_labels_for_postprocessing = None
        if validation_participants:
            validation_features_raw, validation_labels_raw, _validation_subjects = _concat_features(feature_sets, validation_participants)
            train_labels_epoch = train_labels_raw
            validation_labels_epoch = validation_labels_raw
            if config.label_shuffle_control:
                train_labels_epoch = _shuffle_labels_within_subjects(train_labels_epoch, train_subjects, seed=config.label_shuffle_seed, context=(test_participant, 1))
                validation_features_for_shuffle, validation_labels_epoch, validation_subjects = _concat_features(feature_sets, validation_participants)
                validation_labels_epoch = _shuffle_labels_within_subjects(validation_labels_epoch, validation_subjects, seed=config.label_shuffle_seed, context=(test_participant, 2))
                validation_features_raw = validation_features_for_shuffle
            classes_epoch = np.asarray(sorted(set(int(value) for value in train_labels_epoch)), dtype=int)
            _pca, train_features_pca, validation_features_pca, _actual_components, _explained = _fit_pca(
                train_features_raw,
                validation_features_raw,
                components_pca=config.components_pca,
                seed=config.seed,
            )
            validation_tuple = (validation_features_pca, validation_labels_epoch)
            _model, fit_metadata = _train_model(
                train_features_pca,
                train_labels_epoch,
                train_subjects,
                classes=classes_epoch,
                subject_ids=train_epoch_participants,
                config=config,
                validation=validation_tuple,
            )
            device = _resolve_device(config.device)
            validation_latent = _predict_latent(_model, validation_features_pca, device=device, batch_size=config.batch_size)
            validation_neural_scores = _predict_scores(_model, validation_features_pca, device=device, batch_size=config.batch_size)
            validation_head_model, validation_head_metadata = _fit_latent_logistic_head(
                _predict_latent(_model, train_features_pca, device=device, batch_size=config.batch_size),
                train_labels_epoch,
                validation_latent,
                validation_labels_epoch,
                classes_epoch,
                config,
                validation_base_scores=validation_neural_scores,
            )
            if validation_head_model is None:
                validation_scores = validation_neural_scores
            else:
                validation_logistic_scores = _logistic_head_score_matrix(
                    validation_head_model,
                    validation_latent,
                    classes_epoch,
                )
                validation_scores = _apply_latent_head_refit_scores(validation_neural_scores, validation_logistic_scores, config, validation_head_metadata)
            score_calibration_bias, score_calibration_metadata = _fit_validation_score_calibration(
                validation_scores, validation_labels_epoch, classes_epoch, config
            )
            validation_scores_for_postprocessing = _apply_score_calibration(validation_scores, score_calibration_bias)
            validation_labels_for_postprocessing = validation_labels_epoch
            selected_epoch = int(fit_metadata.get("best_epoch", config.epochs))
            fit_metadata = {**fit_metadata, **validation_head_metadata, **score_calibration_metadata}

        final_train_features_raw, final_train_labels, final_train_subjects = _concat_features(feature_sets, source_participants)
        if config.label_shuffle_control:
            final_train_labels = _shuffle_labels_within_subjects(final_train_labels, final_train_subjects, seed=config.label_shuffle_seed, context=(test_participant, 3))
        test_features_raw, test_labels, _test_subjects = _concat_features(feature_sets, (test_participant,))
        classes = np.asarray(sorted(set(int(value) for value in final_train_labels)), dtype=int)
        _final_pca, final_train_features_pca, test_features_pca, actual_components, explained = _fit_pca(
            final_train_features_raw,
            test_features_raw,
            components_pca=config.components_pca,
            seed=config.seed,
        )
        device = _resolve_device(config.device)
        final_epochs = _final_refit_epochs(selected_epoch, config)
        final_training_epochs = final_epochs if config.refit_all_sources else config.epochs
        final_seeds = _effective_ensemble_seeds(config)
        final_score_matrices = []
        final_source_score_matrices = []
        final_epoch_values = []
        for final_seed in final_seeds:
            final_config = replace(config, seed=int(final_seed))
            final_model, final_fit_metadata = _train_model(
                final_train_features_pca,
                final_train_labels,
                final_train_subjects,
                classes=classes,
                subject_ids=source_participants,
                config=final_config,
                validation=None,
                max_epochs=final_training_epochs,
            )
            final_epoch_values.append(int(final_fit_metadata.get("best_epoch", final_training_epochs)))
            selected_head_c = fit_metadata.get("latent_head_refit_selected_c", np.nan)
            selected_head_alpha = fit_metadata.get("latent_head_refit_selected_blend_alpha", np.nan)
            final_head_model, final_head_metadata = _fit_latent_logistic_head(
                _predict_latent(final_model, final_train_features_pca, device=device, batch_size=config.batch_size),
                final_train_labels,
                None,
                None,
                classes,
                final_config,
                selected_c=float(selected_head_c) if np.isfinite(float(selected_head_c)) else None,
                selected_blend_alpha=float(selected_head_alpha) if np.isfinite(float(selected_head_alpha)) else None,
            )
            if fit_metadata.get("latent_head_refit_status") in {None, "not_requested"}:
                fit_metadata = {**fit_metadata, **final_head_metadata}
            final_neural_scores = _predict_scores(final_model, test_features_pca, device=device, batch_size=config.batch_size)
            if final_head_model is None:
                final_score_matrices.append(final_neural_scores)
            else:
                final_logistic_scores = _logistic_head_score_matrix(
                    final_head_model,
                    _predict_latent(final_model, test_features_pca, device=device, batch_size=config.batch_size),
                    classes,
                )
                final_score_matrices.append(_apply_latent_head_refit_scores(final_neural_scores, final_logistic_scores, final_config, fit_metadata))
                fit_metadata.setdefault("latent_head_refit_final_status", final_head_metadata.get("latent_head_refit_status", "ok"))
            if config.score_calibration_final_refit:
                final_source_neural_scores = _predict_scores(final_model, final_train_features_pca, device=device, batch_size=config.batch_size)
                if final_head_model is None:
                    final_source_score_matrices.append(final_source_neural_scores)
                else:
                    final_source_logistic_scores = _logistic_head_score_matrix(
                        final_head_model,
                        _predict_latent(final_model, final_train_features_pca, device=device, batch_size=config.batch_size),
                        classes,
                    )
                    final_source_score_matrices.append(_apply_latent_head_refit_scores(final_source_neural_scores, final_source_logistic_scores, final_config, fit_metadata))
        raw_scores = np.mean(np.stack(final_score_matrices, axis=0), axis=0)
        if config.score_calibration_final_refit and final_source_score_matrices:
            raw_source_scores = np.mean(np.stack(final_source_score_matrices, axis=0), axis=0)
            score_calibration_bias, score_calibration_metadata = _refit_score_calibration_on_source_train(
                score_calibration_bias,
                score_calibration_metadata,
                raw_source_scores,
                final_train_labels,
                classes,
                config,
            )
            fit_metadata = {**fit_metadata, **score_calibration_metadata}
        scores = _apply_score_calibration(raw_scores, score_calibration_bias)
        fit_metadata = {
            **fit_metadata,
            "best_epoch": selected_epoch,
            "final_epochs": int(final_training_epochs),
            "latent_score_ensemble_size": len(final_seeds),
            "latent_score_ensemble_seeds": _format_seed_sequence(final_seeds),
            "latent_score_ensemble_final_epochs": _format_seed_sequence(final_epoch_values),
        }
        predicted_labels, postprocessing_metadata = _postprocess_predictions(
            scores,
            classes,
            final_train_labels,
            config,
            validation_scores=validation_scores_for_postprocessing,
            validation_labels=validation_labels_for_postprocessing,
        )
        fit_metadata = {**fit_metadata, **postprocessing_metadata}
        outer_row = _outer_row(
            test_participant=test_participant,
            train_participants=source_participants,
            true_labels=test_labels,
            predicted_labels=predicted_labels,
            scores=scores,
            classes=classes,
            config=config,
            pca_components=actual_components,
            pca_explained_variance_percent=explained,
            fit_metadata=fit_metadata,
            inner_validation_participants=validation_participants,
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(
            _prediction_rows(
                test_participant=test_participant,
                true_labels=test_labels,
                predicted_labels=predicted_labels,
                scores=scores,
                classes=classes,
                config=config,
                pca_components=actual_components,
                pca_explained_variance_percent=explained,
            )
        )
        if progress is not None:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f} best_epoch={selected_epoch}")

    confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": _group_summary(outer_rows, config),
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": summarize_cross_subject_confusion_pairs(prediction_rows),
    }


def export_latent_autoencoder_loso(  # pylint: disable=too-many-arguments
    data_folder,
    participants: Sequence[int],
    *,
    outer_participants: Sequence[int] | None = None,
    config: LatentAutoencoderConfig | None = None,
    outer_output_path,
    group_summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    progress=None,
) -> dict[str, list[dict]]:
    """Run latent autoencoder LOSO and write CSV artifacts."""

    artifacts = evaluate_latent_autoencoder_loso(data_folder, participants, outer_participants=outer_participants, config=config, progress=progress)
    write_alpha_metrics_csv(artifacts["outer"], outer_output_path)
    if group_summary_output_path:
        write_alpha_metrics_csv(artifacts["group_summary"], group_summary_output_path)
    if predictions_output_path:
        write_alpha_metrics_csv(artifacts["predictions"], predictions_output_path)
    if confusion_output_path:
        write_alpha_metrics_csv(artifacts["confusion"], confusion_output_path)
    if per_stimulus_output_path:
        write_alpha_metrics_csv(artifacts["per_stimulus"], per_stimulus_output_path)
    if confusion_pairs_output_path:
        write_alpha_metrics_csv(artifacts["confusion_pairs"], confusion_pairs_output_path)
    return artifacts


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Experimental source-only LOSO latent autoencoder for BUSH-MEG stimulus decoding.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_LATENT_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Optional held-out participants to evaluate; defaults to all participants.")
    parser.add_argument("--window-center", type=float, default=DEFAULT_LATENT_WINDOW_CENTER)
    parser.add_argument("--window-size", type=float, default=DEFAULT_LATENT_WINDOW_SIZE)
    parser.add_argument("--baseline-window", type=_parse_time_window, default=DEFAULT_LATENT_BASELINE_WINDOW)
    parser.add_argument("--feature-mode", default="sensor_flat")
    parser.add_argument("--normalization", default="subject_baseline_whiten")
    parser.add_argument("--components-pca", type=int, default=DEFAULT_LATENT_COMPONENTS_PCA)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument(
        "--latent-training-preset",
        choices=LATENT_TRAINING_PRESET_CHOICES,
        default=DEFAULT_LATENT_TRAINING_PRESET,
        help=(
            "Named latent-AE preset. anti_collapse_train enables source-only regularizers/selection "
            "for class-collapse control; anti_collapse_calibrated also enables guarded calibration/postprocessing."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_LATENT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        help=(
            "Denoising dropout applied to source-PCA features before the shared encoder. "
            "This preserves inference behavior when set to 0; small values such as 0.03-0.10 "
            "can regularize the latent representation against source-subject/sensor-feature overfit."
        ),
    )
    parser.add_argument("--reconstruction-weight", type=float, default=DEFAULT_LATENT_RECONSTRUCTION_WEIGHT)
    parser.add_argument(
        "--subject-adversary-weight",
        type=float,
        default=0.0,
        help=(
            "Gradient-reversal subject-adversary weight; 0 disables it. "
            "Use small values such as 0.01 or 0.03 to discourage subject-specific latent coding."
        ),
    )
    parser.add_argument("--prediction-balance-weight", type=float, default=0.0)
    parser.add_argument(
        "--prediction-balance-target-smoothing",
        type=float,
        default=1.0,
        help="0=batch label histogram, 1=uniform target distribution.",
    )
    parser.add_argument(
        "--prediction-balance-temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature for prediction-balance regularization. Values below 1, e.g. 0.25-0.5, "
            "make the balance penalty more sensitive to hard-argmax class collapse."
        ),
    )
    parser.add_argument(
        "--logit-mean-center-weight",
        type=float,
        default=0.0,
        help=(
            "Optional source-only regularizer that penalizes minibatch-level class-logit offsets. "
            "Small values such as 0.001 or 0.003 can reduce hard prediction collapse without "
            "forcing a balanced assignment at inference."
        ),
    )
    parser.add_argument(
        "--class-bias-l2-weight",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on the latent classifier bias vector.  Source labels are balanced, "
            "so small values such as 0.001 or 0.003 discourage persistent class-prior offsets "
            "without constraining trial-specific evidence."
        ),
    )
    parser.add_argument(
        "--confidence-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Optional entropy confidence penalty on training logits. Small values such as 0.001 "
            "or 0.003 discourage early over-confident source memorization."
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing for latent AE training; useful for reducing overconfident class collapse.",
    )
    parser.add_argument(
        "--focal-loss-gamma",
        type=float,
        default=0.0,
        help=(
            "Optional focal-loss gamma for latent AE classification. 0 preserves the existing "
            "class-balanced cross-entropy; try 1.0 or 2.0 to emphasize hard source trials/classes."
        ),
    )
    parser.add_argument(
        "--margin-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional true-class-vs-best-negative margin loss weight for latent AE training. "
            "Try small values such as 0.02 or 0.05."
        ),
    )
    parser.add_argument(
        "--margin-loss-value",
        type=float,
        default=1.0,
        help="Required logit margin between the true class and the strongest competing class.",
    )
    parser.add_argument(
        "--soft-macro-recall-weight",
        type=float,
        default=0.0,
        help=(
            "Optional differentiable macro-recall / balanced-accuracy surrogate weight. "
            "Small values such as 0.05 or 0.10 can discourage zero-recall classes."
        ),
    )
    parser.add_argument(
        "--soft-worst-class-recall-weight",
        type=float,
        default=0.0,
        help=(
            "Optional differentiable worst-class soft-recall surrogate weight. "
            "Try small values such as 0.02 or 0.05 when per-class recall collapses."
        ),
    )
    parser.add_argument(
        "--supervised-contrastive-weight",
        type=float,
        default=0.0,
        help=(
            "Optional source-only supervised contrastive latent loss.  Values around 0.01-0.10 "
            "encourage same-class source trials to share latent structure across subjects."
        ),
    )
    parser.add_argument(
        "--supervised-contrastive-temperature",
        type=float,
        default=0.20,
        help="Temperature for the optional supervised contrastive latent loss.",
    )
    parser.add_argument(
        "--balanced-batch-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interleave classes within each training epoch so minibatches are class-diverse.",
    )
    parser.add_argument(
        "--subject-class-balanced-batch-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Interleave source participant/class buckets within each training epoch. "
            "This is stricter than --balanced-batch-sampling and keeps minibatches diverse "
            "across both subjects and classes."
        ),
    )
    parser.add_argument(
        "--validation-prediction-balance-weight",
        type=float,
        default=0.0,
        help=(
            "Optional early-stopping penalty for class-prediction imbalance on "
            "source validation subjects; 0 preserves balanced-accuracy selection."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-source-count", type=int, default=2)
    parser.add_argument(
        "--validation-source-strategy",
        choices=("tail", "head", "spread", "rotating", "round_robin", "seeded_random"),
        default=DEFAULT_LATENT_VALIDATION_SOURCE_STRATEGY,
        help=(
            "How to choose source participants for early stopping/calibration validation. "
            "The default rotates with the held-out participant so a full LOSO run "
            "does not always early-stop on the same source subjects."
        ),
    )
    parser.add_argument(
        "--validation-selection-metric",
        choices=("balanced_accuracy", "balanced_top2_top3_rank", "balanced_top2_top3_rank_balance"),
        default="balanced_accuracy",
        help="Source-validation metric used for epoch selection and early stopping.",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--refit-all-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--final-epoch-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the selected validation epoch for the final all-source refit.",
    )
    parser.add_argument("--final-min-epochs", type=int, default=0, help="Minimum epochs for the final all-source refit.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ensemble-seeds",
        type=_parse_int_sequence,
        default=(),
        help=(
            "Optional comma-separated final-refit seeds. When provided, the command trains one final latent model "
            "per seed and averages their class scores. Validation/epoch selection still uses --seed."
        ),
    )
    parser.add_argument(
        "--latent-head-refit",
        choices=LATENT_HEAD_REFIT_CHOICES,
        default="none",
        help=(
            "Optional source-only refit of the final classifier on frozen encoder latents. "
            "source_logistic uses the first C value; validation_selected_source_logistic selects C on source-validation subjects; "
            "validation_selected_source_logistic_blend also source-validation-selects a neural/logistic score blend alpha."
        ),
    )
    parser.add_argument(
        "--latent-head-refit-c-values",
        type=_parse_float_sequence,
        default=(0.1, 0.3, 1.0, 3.0),
        help="Candidate C values for the optional frozen-latent logistic head refit.",
    )
    parser.add_argument(
        "--latent-head-refit-blend-alphas",
        type=_parse_float_sequence,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Candidate neural/logistic blend alphas for validation_selected_source_logistic_blend.",
    )
    parser.add_argument(
        "--latent-head-refit-selection-metric",
        choices=("balanced_accuracy", "balanced_top2_top3_rank", "balanced_top2_top3_rank_balance"),
        default="balanced_accuracy",
        help="Source-validation metric used to select C for validation_selected_source_logistic.",
    )
    parser.add_argument("--chance-classes", type=int, default=16)
    parser.add_argument("--label-shuffle-control", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
    parser.add_argument(
        "--score-calibration",
        choices=(
            "none",
            "validation_class_bias",
            "validation_class_bias_guarded",
            "validation_prediction_bias",
            "validation_argmax_class_bias",
            "validation_argmax_class_bias_guarded",
            "validation_temperature_class_bias_guarded",
            "validation_temperature_argmax_class_bias_guarded",
            "validation_rank_prior_bias_guarded",
            "validation_confusion_blend",
            "validation_logistic_stack",
            "validation_logistic_stack_guarded",
            "validation_vector_bias",
            "validation_vector_bias_guarded",
            "validation_class_zscore",
            "validation_class_zscore_guarded",
            "validation_score_standardize",
            "validation_score_standardize_guarded",
            "validation_selected_guarded",
        ),
        default="none",
        help="Optional source-validation-only logit calibration for latent AE predictions.",
    )
    parser.add_argument(
        "--score-calibration-alphas",
        type=_parse_float_sequence,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help="Candidate strengths for validation_class_bias calibration.",
    )
    parser.add_argument(
        "--score-calibration-logistic-c-values",
        type=_parse_float_sequence,
        default=(0.03, 0.1, 0.3, 1.0),
        help="Candidate inverse-regularization strengths for validation_logistic_stack calibration.",
    )
    parser.add_argument(
        "--score-calibration-temperatures",
        type=_parse_float_sequence,
        default=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        help="Candidate logit temperatures for validation_temperature_* score calibration.",
    )
    parser.add_argument("--score-calibration-smoothing", type=float, default=1.0)
    parser.add_argument(
        "--score-calibration-confusion-smoothing",
        type=float,
        default=4.0,
        help=(
            "Identity pseudo-counts per class for validation_confusion_blend; "
            "larger values keep the source-validation confusion map more conservative."
        ),
    )
    parser.add_argument(
        "--score-calibration-selection-metric",
        choices=("balanced_accuracy", "balanced_top2_top3_rank", "balanced_top2_top3_rank_balance"),
        default="balanced_accuracy",
        help=(
            "Source-validation objective used by validation_class_bias_guarded. "
            "validation_class_bias keeps the legacy balanced-accuracy objective."
        ),
    )
    parser.add_argument(
        "--score-calibration-guard-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed validation balanced-accuracy drop for validation_class_bias_guarded; "
            "0 enforces no validation balanced-accuracy regression."
        ),
    )
    parser.add_argument(
        "--score-calibration-vector-steps",
        type=_parse_float_sequence,
        default=(0.5, 0.25, 0.125),
        help=(
            "Greedy coordinate-search step sizes for validation_vector_bias calibration. "
            "The bias vector is fitted only on source-validation subjects."
        ),
    )
    parser.add_argument(
        "--score-calibration-vector-rounds",
        type=int,
        default=2,
        help="Maximum coordinate-search passes for validation_vector_bias calibration.",
    )
    parser.add_argument(
        "--score-calibration-vector-l2",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on validation_vector_bias magnitude during source-validation search; "
            "0 disables the penalty."
        ),
    )
    parser.add_argument(
        "--score-calibration-final-refit",
        action="store_true",
        help=(
            "After source-validation selects a score-calibration method, refit the selected calibration "
            "on final source-training scores before applying it to the held-out participant."
        ),
    )
    parser.add_argument(
        "--prediction-postprocessing",
        choices=(
            "none",
            "source_prior_balanced_assignment",
            "source_prior_soft_balanced_assignment",
            "validation_guarded_source_prior_balanced_assignment",
            "validation_guarded_source_prior_soft_balanced_assignment",
            "validation_guarded_shrunk_source_prior_balanced_assignment",
            "validation_selected_balanced_assignment",
        ),
        default="none",
        help=(
            "Optional source-prior quota assignment over the held-out batch. "
            "Soft variants blend argmax counts with the source prior; validation_guarded variants "
            "apply the assignment only when source validation does not regress."
        ),
    )
    parser.add_argument(
        "--prediction-postprocessing-selection-metric",
        choices=("balanced_accuracy", "balanced_top2_top3_rank", "balanced_top2_top3_rank_balance"),
        default="balanced_accuracy",
        help=(
            "Source-validation objective used by validation_selected_balanced_assignment. "
            "Use balanced_top2_top3_rank_balance when selecting among assignment candidates."
        ),
    )
    parser.add_argument(
        "--prediction-postprocessing-shrinkage-alphas",
        type=_parse_float_sequence,
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
        help=(
            "Candidate shrinkage strengths for validation_guarded_shrunk_source_prior_balanced_assignment. "
            "0 preserves the argmax prediction histogram; 1 uses the full source-label prior."
        ),
    )
    parser.add_argument(
        "--prediction-postprocessing-margin-thresholds",
        type=_parse_float_sequence,
        default=(0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
        help=(
            "Candidate top-1/top-2 score-margin thresholds for validation_selected_balanced_assignment. "
            "High-margin argmax predictions are kept fixed; lower-margin rows are quota-assigned."
        ),
    )
    parser.add_argument(
        "--prediction-postprocessing-guard-tolerance",
        type=float,
        default=0.0,
        help="Allowed validation balanced-accuracy drop for validation_guarded_source_prior_balanced_assignment.",
    )
    parser.add_argument(
        "--prediction-postprocessing-quota-strength",
        type=float,
        default=1.0,
        help=(
            "Blend strength for source_prior_soft_balanced_assignment. 0 keeps the model's argmax "
            "prediction counts, 1 uses exact source-prior quotas, and intermediate values partially "
            "correct class-collapse while avoiding a fully forced balanced assignment."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device string.")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--outer-output", default="outputs/latent_autoencoder_outer.csv")
    parser.add_argument("--summary-output", default="outputs/latent_autoencoder_group_summary.csv")
    parser.add_argument("--predictions-output", default="outputs/latent_autoencoder_predictions.csv")
    parser.add_argument("--confusion-output", default="outputs/latent_autoencoder_confusion.csv")
    parser.add_argument("--per-stimulus-output", default="outputs/latent_autoencoder_per_stimulus.csv")
    parser.add_argument("--confusion-pairs-output", default="outputs/latent_autoencoder_confusion_pairs.csv")
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog)
    args = parser.parse_args(normalize_argv(argv))
    config = LatentAutoencoderConfig(
        window_center=args.window_center,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_mode=args.feature_mode,
        normalization=args.normalization,
        training_preset=args.latent_training_preset,
        components_pca=args.components_pca,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        reconstruction_weight=args.reconstruction_weight,
        subject_adversary_weight=args.subject_adversary_weight,
        prediction_balance_weight=args.prediction_balance_weight,
        prediction_balance_target_smoothing=args.prediction_balance_target_smoothing,
        prediction_balance_temperature=args.prediction_balance_temperature,
        logit_mean_center_weight=args.logit_mean_center_weight,
        class_bias_l2_weight=args.class_bias_l2_weight,
        confidence_penalty_weight=args.confidence_penalty_weight,
        label_smoothing=args.label_smoothing,
        focal_loss_gamma=args.focal_loss_gamma,
        margin_loss_weight=args.margin_loss_weight,
        margin_loss_value=args.margin_loss_value,
        soft_macro_recall_weight=args.soft_macro_recall_weight,
        soft_worst_class_recall_weight=args.soft_worst_class_recall_weight,
        supervised_contrastive_weight=args.supervised_contrastive_weight,
        supervised_contrastive_temperature=args.supervised_contrastive_temperature,
        balanced_batch_sampling=bool(args.balanced_batch_sampling),
        subject_class_balanced_batch_sampling=bool(args.subject_class_balanced_batch_sampling),
        validation_prediction_balance_weight=args.validation_prediction_balance_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_source_count=args.validation_source_count,
        validation_source_strategy=args.validation_source_strategy,
        validation_selection_metric=args.validation_selection_metric,
        patience=args.patience,
        refit_all_sources=bool(args.refit_all_sources),
        final_epoch_multiplier=args.final_epoch_multiplier,
        final_min_epochs=args.final_min_epochs,
        seed=args.seed,
        ensemble_seeds=args.ensemble_seeds,
        chance_classes=args.chance_classes,
        latent_head_refit=args.latent_head_refit,
        latent_head_refit_c_values=args.latent_head_refit_c_values,
        latent_head_refit_blend_alphas=args.latent_head_refit_blend_alphas,
        latent_head_refit_selection_metric=args.latent_head_refit_selection_metric,
        label_shuffle_control=bool(args.label_shuffle_control),
        label_shuffle_seed=args.label_shuffle_seed,
        score_calibration=args.score_calibration,
        score_calibration_alphas=args.score_calibration_alphas,
        score_calibration_temperatures=args.score_calibration_temperatures,
        score_calibration_logistic_c_values=args.score_calibration_logistic_c_values,
        score_calibration_smoothing=args.score_calibration_smoothing,
        score_calibration_confusion_smoothing=args.score_calibration_confusion_smoothing,
        score_calibration_selection_metric=args.score_calibration_selection_metric,
        score_calibration_guard_tolerance=args.score_calibration_guard_tolerance,
        score_calibration_vector_steps=args.score_calibration_vector_steps,
        score_calibration_vector_rounds=args.score_calibration_vector_rounds,
        score_calibration_vector_l2=args.score_calibration_vector_l2,
        score_calibration_final_refit=bool(args.score_calibration_final_refit),
        prediction_postprocessing=args.prediction_postprocessing,
        prediction_postprocessing_guard_tolerance=args.prediction_postprocessing_guard_tolerance,
        prediction_postprocessing_selection_metric=args.prediction_postprocessing_selection_metric,
        prediction_postprocessing_quota_strength=args.prediction_postprocessing_quota_strength,
        prediction_postprocessing_margin_thresholds=args.prediction_postprocessing_margin_thresholds,
        prediction_postprocessing_shrinkage_alphas=args.prediction_postprocessing_shrinkage_alphas,
        device=args.device,
        num_threads=args.num_threads,
    )
    config = _apply_latent_training_preset(config, args.latent_training_preset)
    for path in (
        args.outer_output,
        args.summary_output,
        args.predictions_output,
        args.confusion_output,
        args.per_stimulus_output,
        args.confusion_pairs_output,
    ):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    export_latent_autoencoder_loso(
        resolve_data_folder(args.data_folder),
        _parse_participants(args.participants),
        outer_participants=_parse_participants(args.outer_participants) if args.outer_participants else None,
        config=config,
        outer_output_path=args.outer_output,
        group_summary_output_path=args.summary_output,
        predictions_output_path=args.predictions_output,
        confusion_output_path=args.confusion_output,
        per_stimulus_output_path=args.per_stimulus_output,
        confusion_pairs_output_path=args.confusion_pairs_output,
        progress=lambda message: print(message, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
