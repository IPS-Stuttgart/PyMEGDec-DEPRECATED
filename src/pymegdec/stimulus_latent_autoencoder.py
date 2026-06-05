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
DEFAULT_LATENT_VALIDATION_SOURCE_STRATEGY = "rotating"


@dataclass(frozen=True)
class LatentAutoencoderConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for fixed source-only latent autoencoder LOSO decoding."""

    window_center: float = DEFAULT_LATENT_WINDOW_CENTER
    window_size: float = DEFAULT_LATENT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_LATENT_BASELINE_WINDOW
    feature_mode: str = "sensor_flat"
    normalization: str = "subject_baseline_whiten"
    components_pca: int = DEFAULT_LATENT_COMPONENTS_PCA
    latent_dim: int = DEFAULT_LATENT_DIM
    hidden_dim: int = DEFAULT_LATENT_HIDDEN_DIM
    dropout: float = 0.10
    reconstruction_weight: float = DEFAULT_LATENT_RECONSTRUCTION_WEIGHT
    subject_adversary_weight: float = 0.0
    prediction_balance_weight: float = 0.0
    prediction_balance_target_smoothing: float = 1.0
    label_smoothing: float = 0.0
    supervised_contrastive_weight: float = 0.0
    supervised_contrastive_temperature: float = 0.20
    balanced_batch_sampling: bool = False
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
    score_calibration: str = "none"
    score_calibration_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    score_calibration_smoothing: float = 1.0
    score_calibration_confusion_smoothing: float = 4.0
    score_calibration_selection_metric: str = "balanced_accuracy"
    score_calibration_guard_tolerance: float = 0.0
    prediction_postprocessing: str = "none"
    prediction_postprocessing_guard_tolerance: float = 0.0
    prediction_postprocessing_shrinkage_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
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
        def __init__(self, *, n_features: int, n_classes: int, subject_ids: Iterable[int], hidden_dim: int, latent_dim: int, dropout: float):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            self.classifier = nn.Linear(latent_dim, n_classes)
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
            latent = self.encoder(features)
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


def _prediction_balance_loss(logits, label_indices, *, target_smoothing: float):
    """Penalize minibatch-level predicted-class collapse."""

    torch, _nn, F = _lazy_torch()
    if int(logits.shape[0]) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    probabilities = F.softmax(logits, dim=1)
    predicted_distribution = probabilities.mean(dim=0)
    label_distribution = (
        F.one_hot(label_indices, num_classes=int(logits.shape[1])).to(dtype=logits.dtype).mean(dim=0)
    )
    uniform_distribution = torch.full_like(predicted_distribution, 1.0 / float(logits.shape[1]))
    smoothing = min(max(float(target_smoothing), 0.0), 1.0)
    target_distribution = (1.0 - smoothing) * label_distribution + smoothing * uniform_distribution
    return F.mse_loss(predicted_distribution, target_distribution)


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
    anchor: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_participants = tuple(int(value) for value in source_participants)
    count = max(0, int(validation_source_count))
    if count == 0 or len(source_participants) <= count + 1:
        return source_participants, tuple()

    n_sources = len(source_participants)
    strategy = str(strategy or "tail")
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
    else:
        raise ValueError(
            "validation_source_strategy must be one of: tail, head, spread, rotating"
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
        if config.balanced_batch_sampling:
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
            class_loss = F.cross_entropy(
                logits,
                yb,
                weight=weights,
                label_smoothing=_bounded_label_smoothing(config.label_smoothing),
            )
            reconstruction_losses = []
            for subject_id in torch.unique(pb).detach().cpu().numpy().tolist():
                mask = pb == int(subject_id)
                if bool(torch.any(mask)):
                    reconstruction = model.reconstruct_subject(int(subject_id), latent[mask])
                    reconstruction_losses.append(F.mse_loss(reconstruction, xb[mask]))
            reconstruction_loss = torch.stack(reconstruction_losses).mean() if reconstruction_losses else torch.zeros((), device=device)
            loss = class_loss + float(config.reconstruction_weight) * reconstruction_loss
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
                )
                loss = loss + float(config.prediction_balance_weight) * balance_loss
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


def _empty_score_calibration_metadata(config: LatentAutoencoderConfig, status: str) -> dict:
    return {
        "score_calibration": config.score_calibration,
        "score_calibration_status": status,
        "score_calibration_prior_source": "",
        "score_calibration_predicted_prior_source": "none",
        "score_calibration_alpha": 0.0,
        "score_calibration_validation_balanced_accuracy": np.nan,
        "score_calibration_uncalibrated_validation_balanced_accuracy": np.nan,
        "score_calibration_selection_metric": config.score_calibration_selection_metric,
        "score_calibration_validation_selection_score": np.nan,
        "score_calibration_uncalibrated_validation_selection_score": np.nan,
        "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
        "score_calibration_bias_min": 0.0,
        "score_calibration_bias_max": 0.0,
        "score_calibration_bias_mean_abs": 0.0,
        "score_calibration_confusion_smoothing": float(config.score_calibration_confusion_smoothing),
        "score_calibration_confusion_map_trace": np.nan,
        "score_calibration_scale_min": np.nan,
        "score_calibration_scale_max": np.nan,
    }


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
        "validation_score_standardize",
        "validation_score_standardize_guarded",
    }
    if config.score_calibration not in supported_calibrations:
        raise ValueError(f"Unsupported latent AE score calibration: {config.score_calibration!r}")
    if validation_scores is None or validation_labels is None or len(validation_labels) == 0:
        return zero_bias, _empty_score_calibration_metadata(config, "no_validation")
    if config.score_calibration == "validation_confusion_blend":
        return _fit_validation_confusion_blend_calibration(validation_scores, validation_labels, classes, config)
    if config.score_calibration in {
        "validation_score_standardize",
        "validation_score_standardize_guarded",
    }:
        return _fit_validation_score_standardization_calibration(validation_scores, validation_labels, classes, config)

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
    }
    return calibrator, metadata


def _apply_score_calibration(scores: np.ndarray, calibration: np.ndarray | dict | None) -> np.ndarray:
    if calibration is None or len(calibration) == 0:
        return np.asarray(scores, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if isinstance(calibration, dict):
        if calibration.get("kind") == "score_standardize":
            alpha = min(max(float(calibration.get("alpha", 0.0)), 0.0), 1.0)
            means = np.asarray(calibration["mean"], dtype=float).reshape(1, -1)
            scales = np.asarray(calibration["scale"], dtype=float).reshape(1, -1)
            scales = np.maximum(scales, 1e-6)
            standardized = (scores - means) / scales
            return (1.0 - alpha) * scores + alpha * standardized
        if calibration.get("kind") != "confusion_blend":
            raise ValueError(f"Unknown score calibration kind: {calibration.get('kind')!r}")
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
            "components_pca": config.components_pca,
            "actual_components_pca": pca_components,
            "pca_explained_variance_percent": pca_explained_variance_percent,
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "seed": config.seed,
            "latent_score_ensemble_size": len(_effective_ensemble_seeds(config)),
            "latent_score_ensemble_seeds": _format_seed_sequence(_effective_ensemble_seeds(config)),
            "reconstruction_weight": config.reconstruction_weight,
            "subject_adversary_weight": config.subject_adversary_weight,
            "prediction_balance_weight": config.prediction_balance_weight,
            "prediction_balance_target_smoothing": config.prediction_balance_target_smoothing,
            "label_smoothing": config.label_smoothing,
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
        "accuracy": accuracy,
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
        "seed": config.seed,
        "latent_score_ensemble_size": int(fit_metadata.get("latent_score_ensemble_size", len(_effective_ensemble_seeds(config)))),
        "latent_score_ensemble_seeds": fit_metadata.get("latent_score_ensemble_seeds", _format_seed_sequence(_effective_ensemble_seeds(config))),
        "latent_score_ensemble_final_epochs": fit_metadata.get("latent_score_ensemble_final_epochs", ""),
        "reconstruction_weight": config.reconstruction_weight,
        "subject_adversary_weight": config.subject_adversary_weight,
        "prediction_balance_weight": config.prediction_balance_weight,
        "prediction_balance_target_smoothing": config.prediction_balance_target_smoothing,
        "label_smoothing": config.label_smoothing,
        "supervised_contrastive_weight": config.supervised_contrastive_weight,
        "supervised_contrastive_temperature": config.supervised_contrastive_temperature,
        "balanced_batch_sampling": config.balanced_batch_sampling,
        "validation_prediction_balance_weight": config.validation_prediction_balance_weight,
        "score_calibration": config.score_calibration,
        "score_calibration_status": fit_metadata.get("score_calibration_status", "unknown"),
        "score_calibration_prior_source": fit_metadata.get("score_calibration_prior_source", ""),
        "score_calibration_predicted_prior_source": fit_metadata.get("score_calibration_predicted_prior_source", "none"),
        "score_calibration_alpha": fit_metadata.get("score_calibration_alpha", np.nan),
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
        "prediction_postprocessing": config.prediction_postprocessing,
        "prediction_postprocessing_status": fit_metadata.get("prediction_postprocessing_status", "not_requested"),
        "prediction_postprocessing_quota_source": fit_metadata.get("prediction_postprocessing_quota_source", "none"),
        "prediction_postprocessing_class_quota_counts": fit_metadata.get("prediction_postprocessing_class_quota_counts", ""),
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
            "label_smoothing": config.label_smoothing,
            "supervised_contrastive_weight": config.supervised_contrastive_weight,
            "supervised_contrastive_temperature": config.supervised_contrastive_temperature,
            "balanced_batch_sampling": config.balanced_batch_sampling,
            "validation_prediction_balance_weight": config.validation_prediction_balance_weight,
            "dropout": config.dropout,
            "score_calibration": config.score_calibration,
            "score_calibration_alphas": ";".join(str(float(alpha)) for alpha in config.score_calibration_alphas),
            "score_calibration_smoothing": config.score_calibration_smoothing,
            "score_calibration_confusion_smoothing": config.score_calibration_confusion_smoothing,
            "score_calibration_selection_metric": config.score_calibration_selection_metric,
            "score_calibration_guard_tolerance": config.score_calibration_guard_tolerance,
            "score_calibration_status_counts": _format_counter(Counter(row.get("score_calibration_status", "unknown") for row in outer_rows)),
            "score_calibration_prior_source_counts": _format_counter(Counter(row.get("score_calibration_prior_source", "") for row in outer_rows)),
            "score_calibration_predicted_prior_source_counts": _format_counter(
                Counter(row.get("score_calibration_predicted_prior_source", "none") for row in outer_rows)
            ),
            "prediction_postprocessing": config.prediction_postprocessing,
            "prediction_postprocessing_status_counts": _format_counter(
                Counter(row.get("prediction_postprocessing_status", "not_requested") for row in outer_rows)
            ),
            "prediction_postprocessing_guard_tolerance": config.prediction_postprocessing_guard_tolerance,
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
    argmax_labels = classes[np.argmax(scores, axis=1)]
    base_metadata = {
        "prediction_postprocessing_quota_source": "none",
        "prediction_postprocessing_class_quota_counts": "",
        "prediction_postprocessing_objective_delta": 0.0,
        "prediction_postprocessing_validation_balanced_accuracy": np.nan,
        "prediction_postprocessing_uncalibrated_validation_balanced_accuracy": np.nan,
        "prediction_postprocessing_validation_objective_delta": np.nan,
        "prediction_postprocessing_guard_tolerance": float(config.prediction_postprocessing_guard_tolerance),
        "prediction_postprocessing_shrinkage_alpha": np.nan,
        "prediction_postprocessing_shrinkage_alphas": ";".join(str(float(alpha)) for alpha in config.prediction_postprocessing_shrinkage_alphas),
    }
    if method == "none":
        return argmax_labels, {
            **base_metadata,
            "prediction_postprocessing_status": "not_requested",
        }
    supported = {
        "source_prior_balanced_assignment",
        "validation_guarded_source_prior_balanced_assignment",
        "validation_guarded_shrunk_source_prior_balanced_assignment",
    }
    if method not in supported:
        raise ValueError(
            "prediction_postprocessing must be one of: none, source_prior_balanced_assignment, "
            "validation_guarded_source_prior_balanced_assignment, "
            "validation_guarded_shrunk_source_prior_balanced_assignment"
        )

    validation_metadata = dict(base_metadata)
    selected_shrinkage_alpha = 1.0
    if method in {"validation_guarded_source_prior_balanced_assignment", "validation_guarded_shrunk_source_prior_balanced_assignment"}:
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
            candidate_rows = []
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
                candidate_rows.append(
                    (
                        validation_balanced,
                        float(validation_objective_delta),
                        float(candidate_alpha),
                        validation_assigned,
                        validation_quotas,
                    )
                )
            candidate_rows.sort(key=lambda row: (row[0], row[1], -abs(row[2])), reverse=True)
            validation_balanced, validation_objective_delta, selected_shrinkage_alpha, validation_assigned, validation_quotas = candidate_rows[0]
            validation_metadata["prediction_postprocessing_shrinkage_alpha"] = float(selected_shrinkage_alpha)
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
            }

    if method == "validation_guarded_shrunk_source_prior_balanced_assignment":
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
        "prediction_postprocessing_quota_source": quota_source,
        "prediction_postprocessing_class_quota_counts": _format_counter(quota_counts),
        "prediction_postprocessing_objective_delta": float(objective_delta),
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
            validation_scores = _predict_scores(_model, validation_features_pca, device=device, batch_size=config.batch_size)
            score_calibration_bias, score_calibration_metadata = _fit_validation_score_calibration(
                validation_scores, validation_labels_epoch, classes_epoch, config
            )
            validation_scores_for_postprocessing = _apply_score_calibration(validation_scores, score_calibration_bias)
            validation_labels_for_postprocessing = validation_labels_epoch
            selected_epoch = int(fit_metadata.get("best_epoch", config.epochs))
            fit_metadata = {**fit_metadata, **score_calibration_metadata}

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
            final_score_matrices.append(
                _predict_scores(final_model, test_features_pca, device=device, batch_size=config.batch_size)
            )
        scores = np.mean(np.stack(final_score_matrices, axis=0), axis=0)
        scores = _apply_score_calibration(scores, score_calibration_bias)
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
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_LATENT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=0.10)
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
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Cross-entropy label smoothing for latent AE training; useful for reducing overconfident class collapse.",
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
        choices=("tail", "head", "spread", "rotating"),
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
            "validation_confusion_blend",
            "validation_score_standardize",
            "validation_score_standardize_guarded",
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
        "--prediction-postprocessing",
        choices=(
            "none",
            "source_prior_balanced_assignment",
            "validation_guarded_source_prior_balanced_assignment",
            "validation_guarded_shrunk_source_prior_balanced_assignment",
        ),
        default="none",
        help=(
            "Optional source-prior balanced assignment over the held-out batch. "
            "The validation_guarded variant applies it only when source validation does not regress."
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
        "--prediction-postprocessing-guard-tolerance",
        type=float,
        default=0.0,
        help="Allowed validation balanced-accuracy drop for validation_guarded_source_prior_balanced_assignment.",
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
        components_pca=args.components_pca,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        reconstruction_weight=args.reconstruction_weight,
        subject_adversary_weight=args.subject_adversary_weight,
        prediction_balance_weight=args.prediction_balance_weight,
        prediction_balance_target_smoothing=args.prediction_balance_target_smoothing,
        label_smoothing=args.label_smoothing,
        supervised_contrastive_weight=args.supervised_contrastive_weight,
        supervised_contrastive_temperature=args.supervised_contrastive_temperature,
        balanced_batch_sampling=bool(args.balanced_batch_sampling),
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
        label_shuffle_control=bool(args.label_shuffle_control),
        label_shuffle_seed=args.label_shuffle_seed,
        score_calibration=args.score_calibration,
        score_calibration_alphas=args.score_calibration_alphas,
        score_calibration_smoothing=args.score_calibration_smoothing,
        score_calibration_confusion_smoothing=args.score_calibration_confusion_smoothing,
        score_calibration_selection_metric=args.score_calibration_selection_metric,
        score_calibration_guard_tolerance=args.score_calibration_guard_tolerance,
        prediction_postprocessing=args.prediction_postprocessing,
        prediction_postprocessing_guard_tolerance=args.prediction_postprocessing_guard_tolerance,
        prediction_postprocessing_shrinkage_alphas=args.prediction_postprocessing_shrinkage_alphas,
        device=args.device,
        num_threads=args.num_threads,
    )
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
