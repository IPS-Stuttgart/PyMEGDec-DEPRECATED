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
from dataclasses import dataclass
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
    epochs: int = 80
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_source_count: int = 2
    patience: int = 12
    refit_all_sources: bool = True
    seed: int = 0
    chance_classes: int = 16
    label_shuffle_control: bool = False
    label_shuffle_seed: int = 0
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


def _parse_participants(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(parse_participant_spec(DEFAULT_LATENT_PARTICIPANTS))
    return tuple(parse_participant_spec(value))


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
            self.decoders = nn.ModuleDict(
                {
                    self._key(subject_id): nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, n_features),
                    )
                    for subject_id in subject_ids
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


def _fit_pca(train_features: np.ndarray, test_features: np.ndarray | None, *, components_pca: int, seed: int):
    n_components = int(min(int(components_pca), train_features.shape[0], train_features.shape[1]))
    if n_components < 1:
        raise ValueError("PCA requires at least one component.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    train_latent_input = pca.fit_transform(train_features).astype(np.float32)
    test_latent_input = None if test_features is None else pca.transform(test_features).astype(np.float32)
    explained = float(100.0 * np.sum(pca.explained_variance_ratio_))
    return pca, train_latent_input, test_latent_input, n_components, explained


def _split_source_participants(source_participants: Sequence[int], validation_source_count: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_participants = tuple(int(value) for value in source_participants)
    count = max(0, int(validation_source_count))
    if count == 0 or len(source_participants) <= count + 1:
        return source_participants, tuple()
    validation = tuple(source_participants[-count:])
    train = tuple(participant for participant in source_participants if participant not in validation)
    return train, validation


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
    epochs_since_improvement = 0
    history = []
    rng = np.random.default_rng(config.seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.tensor(rng.permutation(train_features.shape[0]), dtype=torch.long, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, int(permutation.shape[0]), int(config.batch_size)):
            batch_index = permutation[start : start + int(config.batch_size)]
            xb = x_tensor[batch_index]
            yb = y_tensor[batch_index]
            pb = p_tensor[batch_index]
            logits, latent = model(xb)
            class_loss = F.cross_entropy(logits, yb, weight=weights)
            reconstruction_losses = []
            for subject_id in torch.unique(pb).detach().cpu().numpy().tolist():
                mask = pb == int(subject_id)
                if bool(torch.any(mask)):
                    reconstruction = model.reconstruct_subject(int(subject_id), latent[mask])
                    reconstruction_losses.append(F.mse_loss(reconstruction, xb[mask]))
            reconstruction_loss = torch.stack(reconstruction_losses).mean() if reconstruction_losses else torch.zeros((), device=device)
            loss = class_loss + float(config.reconstruction_weight) * reconstruction_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1

        validation_balanced = np.nan
        if validation is not None:
            validation_features, validation_labels = validation
            validation_scores = _predict_scores(model, validation_features, device=device, batch_size=config.batch_size)
            validation_pred = classes[np.argmax(validation_scores, axis=1)]
            validation_balanced = float(balanced_accuracy_score(validation_labels, validation_pred))
            if validation_balanced > best_validation_balanced + 1e-8:
                best_validation_balanced = validation_balanced
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

        history.append({"epoch": epoch, "loss": epoch_loss / max(1, batches), "validation_balanced_accuracy": validation_balanced})

    model.load_state_dict(best_state)
    return model, {"best_epoch": int(best_epoch), "best_validation_balanced_accuracy": float(best_validation_balanced), "history": history}


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
            "reconstruction_weight": config.reconstruction_weight,
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
        "reconstruction_weight": config.reconstruction_weight,
        "epochs_requested": config.epochs,
        "best_epoch": fit_metadata.get("best_epoch", np.nan),
        "best_validation_balanced_accuracy": fit_metadata.get("best_validation_balanced_accuracy", np.nan),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "validation_source_count": config.validation_source_count,
        "refit_all_sources": config.refit_all_sources,
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
            "reconstruction_weight": config.reconstruction_weight,
            "dropout": config.dropout,
            "epochs_requested": config.epochs,
            "validation_source_count": config.validation_source_count,
            "refit_all_sources": config.refit_all_sources,
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
            "actual_components_pca_counts": _format_counter(Counter(int(row["actual_components_pca"]) for row in outer_rows)),
        }
    ]


def _format_counter(counter: Counter) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter, key=lambda value: str(value)))


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
        train_epoch_participants, validation_participants = _split_source_participants(source_participants, config.validation_source_count)
        train_features_raw, train_labels_raw, train_subjects = _concat_features(feature_sets, train_epoch_participants)
        validation_tuple = None
        selected_epoch = config.epochs
        fit_metadata: dict = {"best_epoch": config.epochs, "best_validation_balanced_accuracy": np.nan}
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
            selected_epoch = int(fit_metadata.get("best_epoch", config.epochs))

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
        final_model, final_fit_metadata = _train_model(
            final_train_features_pca,
            final_train_labels,
            final_train_subjects,
            classes=classes,
            subject_ids=source_participants,
            config=config,
            validation=None,
            max_epochs=selected_epoch if config.refit_all_sources else config.epochs,
        )
        fit_metadata = {**fit_metadata, "best_epoch": selected_epoch, "final_epochs": final_fit_metadata.get("best_epoch", selected_epoch)}
        device = _resolve_device(config.device)
        scores = _predict_scores(final_model, test_features_pca, device=device, batch_size=config.batch_size)
        predicted_labels = classes[np.argmax(scores, axis=1)]
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
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-source-count", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--refit-all-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chance-classes", type=int, default=16)
    parser.add_argument("--label-shuffle-control", action="store_true")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_source_count=args.validation_source_count,
        patience=args.patience,
        refit_all_sources=bool(args.refit_all_sources),
        seed=args.seed,
        chance_classes=args.chance_classes,
        label_shuffle_control=bool(args.label_shuffle_control),
        label_shuffle_seed=args.label_shuffle_seed,
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
