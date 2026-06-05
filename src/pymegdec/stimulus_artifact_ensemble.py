"""Combine existing nested-matrix prediction artifacts without refitting models."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pymegdec.alpha_metrics import write_alpha_metrics_csv

PREDICTION_FILE_CANDIDATES = ("nested_matrix_predictions.csv",)
DEFAULT_KEY_COLUMNS = ("test_participant", "test_trial_index", "true_label")
CLASS_SCORE_RE = re.compile(r"^(?:score|prob)_class_(-?\d+)$")
CLASS_RANK_RE = re.compile(r"^rank_class_(-?\d+)$")
STIMULUS_SCORE_RE = re.compile(r"^(?:score|prob)_([1-9]\d*)$")
STIMULUS_RANK_RE = re.compile(r"^rank_([1-9]\d*)$")
CLASS_SCORE_PATTERNS = ((CLASS_SCORE_RE, 0), (STIMULUS_SCORE_RE, -1))
CLASS_RANK_PATTERNS = ((CLASS_RANK_RE, 0), (STIMULUS_RANK_RE, -1))
ARTIFACT_SCORE_NORMALIZATION_CHOICES = ("raw", "rank_softmax", "z_softmax")
ARTIFACT_AGGREGATION_MODE_CHOICES = (
    "auto",
    "hard_vote",
    "mean_score",
    "mean_rank",
    "borda",
    "score_tiebreak_first_source",
)


@dataclass(frozen=True)
class PredictionSource:
    """Named trial-level prediction table loaded from an artifact."""

    name: str
    path: Path
    rows: list[dict[str, str]]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_with_consistent_fields(rows: Iterable[dict]) -> list[dict]:
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def write_csv_rows(path: Path, rows: Iterable[dict]) -> None:
    write_alpha_metrics_csv(_rows_with_consistent_fields(rows), path)


def resolve_prediction_csv(path: Path) -> Path:
    """Resolve an artifact directory or CSV path to a prediction CSV."""

    if path.is_file():
        return path
    for name in PREDICTION_FILE_CANDIDATES:
        candidate = path / name
        if candidate.exists():
            return candidate
    matches = sorted(path.rglob("*_predictions.csv"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No *_predictions.csv file found below {path}.")
    preferred = [match for match in matches if match.name == "nested_matrix_predictions.csv"]
    if preferred:
        return preferred[0]
    raise ValueError(f"Found multiple prediction CSV files below {path}; pass one explicitly.")


def load_prediction_source(spec: str) -> PredictionSource:
    """Load a ``name=path`` source specification."""

    if "=" not in spec:
        raise ValueError(f"Source must use name=path syntax, got {spec!r}.")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Source name is empty in {spec!r}.")
    path = resolve_prediction_csv(Path(raw_path.strip()))
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"Source {name!r} has no prediction rows: {path}")
    return PredictionSource(name=name, path=path, rows=rows)


def parse_ensemble_spec(spec: str) -> tuple[str, tuple[str, ...]]:
    """Return ``(ensemble_name, source_names)`` from ``name=a,b,c``."""

    if "=" not in spec:
        raise ValueError(f"Ensemble must use name=source_a,source_b syntax, got {spec!r}.")
    name, raw_sources = spec.split("=", 1)
    source_names = tuple(token.strip().split(":", 1)[0] for token in raw_sources.split(",") if token.strip())
    if not name.strip() or not source_names:
        raise ValueError(f"Invalid ensemble specification: {spec!r}.")
    return name.strip(), source_names


def parse_weighted_ensemble_spec(spec: str) -> tuple[str, tuple[str, ...], tuple[float, ...] | None]:
    """Return ``(ensemble_name, source_names, weights)`` from a CLI ensemble spec.

    Source weights are optional and use ``name=source_a:0.8,source_b:0.2``.
    If no weights are present, source scores are averaged uniformly.
    """

    if "=" not in spec:
        raise ValueError(f"Ensemble must use name=source_a,source_b syntax, got {spec!r}.")
    name, raw_sources = spec.split("=", 1)
    source_names: list[str] = []
    weights: list[float | None] = []
    for token in (part.strip() for part in raw_sources.split(",") if part.strip()):
        if ":" in token:
            source_name, raw_weight = token.rsplit(":", 1)
            source_names.append(source_name.strip())
            weights.append(float(raw_weight))
        else:
            source_names.append(token)
            weights.append(None)
    if not name.strip() or not source_names:
        raise ValueError(f"Invalid ensemble specification: {spec!r}.")
    if any(weight is not None for weight in weights):
        if any(weight is None for weight in weights):
            raise ValueError("Either provide a weight for every source in an ensemble or no weights.")
        return name.strip(), tuple(source_names), tuple(float(weight) for weight in weights if weight is not None)
    return name.strip(), tuple(source_names), None


def _string_key(row: dict[str, str], columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in columns)


def _index_rows(source: PredictionSource, key_columns: Sequence[str]) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row_number, row in enumerate(source.rows, start=2):
        key = _string_key(row, key_columns)
        if any(value == "" for value in key):
            raise ValueError(f"{source.name} row {row_number} is missing one of key columns {', '.join(key_columns)}.")
        if key in indexed:
            raise ValueError(f"{source.name} has duplicate prediction key {key}.")
        indexed[key] = row
    return indexed


def _to_int(value: object, *, field: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Missing integer value for {field}.")
    return int(float(text))


def _to_float(value: object) -> float:
    return float(str(value).strip())


def _label_from_row(row: dict[str, str], *, label_column: str, stimulus_column: str, field: str) -> int:
    raw_label = row.get(label_column)
    if raw_label is not None and str(raw_label).strip() != "":
        return _to_int(raw_label, field=field)
    raw_stimulus = row.get(stimulus_column)
    if raw_stimulus is None or str(raw_stimulus).strip() == "":
        raise ValueError(f"Missing {label_column} or {stimulus_column}.")
    return _to_int(raw_stimulus, field=stimulus_column) - 1


def _source_rank(row: dict[str, str]) -> float | None:
    value = str(row.get("true_label_rank", "")).strip()
    if value == "":
        return None
    try:
        rank = float(value)
    except ValueError:
        return None
    return rank if math.isfinite(rank) else None


def _all_labels(rows: Iterable[dict[str, str]]) -> list[int]:
    labels: set[int] = set()
    for row in rows:
        for label_column, stimulus_column, field in (
            ("true_label", "true_stimulus", "true_label"),
            ("predicted_label", "predicted_stimulus", "predicted_label"),
        ):
            try:
                labels.add(_label_from_row(row, label_column=label_column, stimulus_column=stimulus_column, field=field))
            except ValueError:
                continue
    return sorted(labels)


def _rank_labels_by_hard_votes(
    source_rows: Sequence[dict[str, str]],
    *,
    class_labels: Sequence[int],
    source_weights: Sequence[float] | None = None,
) -> list[int]:
    predictions = [
        _label_from_row(row, label_column="predicted_label", stimulus_column="predicted_stimulus", field="predicted_label")
        for row in source_rows
    ]
    weights = _normalized_source_weights(source_weights, len(source_rows))
    votes: dict[int, float] = defaultdict(float)
    for prediction, weight in zip(predictions, weights, strict=True):
        votes[prediction] += weight
    first_source_index: dict[int, int] = {}
    for index, predicted in enumerate(predictions):
        first_source_index.setdefault(predicted, index)
    return sorted(
        class_labels,
        key=lambda label: (
            -votes.get(label, 0),
            first_source_index.get(label, len(source_rows)),
            label,
        ),
    )


def _normalized_source_weights(weights: Sequence[float] | None, n_sources: int) -> tuple[float, ...]:
    if n_sources <= 0:
        raise ValueError("At least one source is required.")
    if weights is None:
        return tuple(1.0 / n_sources for _ in range(n_sources))
    values = tuple(float(weight) for weight in weights)
    if len(values) != n_sources:
        raise ValueError(f"Expected {n_sources} source weight(s), got {len(values)}.")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("Source weights must be finite non-negative values.")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("At least one source weight must be positive.")
    return tuple(value / total for value in values)


def _normalize_score_values(values: Sequence[float], score_normalization: str) -> list[float]:
    normalized = _normalize_artifact_score_normalization(score_normalization)
    finite_values = [float(value) for value in values]
    if normalized == "raw":
        return finite_values

    if normalized == "rank_softmax":
        order = sorted(range(len(finite_values)), key=lambda index: (-finite_values[index], index))
        ranks = [0.0] * len(finite_values)
        for rank, index in enumerate(order):
            ranks[index] = -float(rank)
        return _softmax(ranks)

    mean = sum(finite_values) / len(finite_values)
    variance = sum((value - mean) ** 2 for value in finite_values) / len(finite_values)
    scale = math.sqrt(variance)
    if scale <= 1e-12 or not math.isfinite(scale):
        return _softmax([0.0 for _ in finite_values])
    return _softmax([(value - mean) / scale for value in finite_values])


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _normalize_artifact_score_normalization(score_normalization: str) -> str:
    normalized = str(score_normalization).strip().lower().replace("-", "_")
    if normalized not in ARTIFACT_SCORE_NORMALIZATION_CHOICES:
        raise ValueError(
            "Artifact score normalization must be one of "
            f"{', '.join(ARTIFACT_SCORE_NORMALIZATION_CHOICES)}."
        )
    return normalized


def _normalize_artifact_aggregation_mode(aggregation_mode: str) -> str:
    normalized = str(aggregation_mode).strip().lower().replace("-", "_")
    aliases = {
        "score": "mean_score",
        "score_mean": "mean_score",
        "class_score_mean": "mean_score",
        "rank": "mean_rank",
        "rank_mean": "mean_rank",
        "class_rank_mean": "mean_rank",
        "class_rank_borda": "borda",
        "hard": "hard_vote",
        "vote": "hard_vote",
        "score_compact_tiebreak": "score_tiebreak_first_source",
        "score_first_source_tiebreak": "score_tiebreak_first_source",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ARTIFACT_AGGREGATION_MODE_CHOICES:
        raise ValueError(
            "Artifact aggregation mode must be one of "
            f"{', '.join(ARTIFACT_AGGREGATION_MODE_CHOICES)}."
        )
    return normalized


def _class_value_columns(rows: Sequence[dict[str, str]], patterns: Sequence[tuple[re.Pattern[str], int]]) -> dict[int, str]:
    common: dict[int, str] | None = None
    for row in rows:
        columns: dict[int, str] = {}
        for column in row:
            if str(row.get(column, "")).strip() == "":
                continue
            for pattern, offset in patterns:
                match = pattern.match(column)
                if match:
                    columns[int(match.group(1)) + offset] = column
                    break
        common = columns if common is None else {label: column for label, column in common.items() if label in columns}
    return common or {}


def _rank_labels_by_scores(
    source_rows: Sequence[dict[str, str]],
    *,
    class_labels: Sequence[int],
    source_weights: Sequence[float] | None = None,
    score_normalization: str = "raw",
    aggregation_mode: str = "auto",
    tie_break_labels: Sequence[int] = (),
) -> tuple[list[int], str] | None:
    mode = _normalize_artifact_aggregation_mode(aggregation_mode)
    if mode in {"auto", "mean_score", "score_tiebreak_first_source"}:
        score_columns = _class_value_columns(source_rows, CLASS_SCORE_PATTERNS)
    else:
        score_columns = {}
    if score_columns:
        normalized = _normalize_artifact_score_normalization(score_normalization)
        weights = _normalized_source_weights(source_weights, len(source_rows))
        scored_labels = [label for label in class_labels if label in score_columns]
        scores: dict[int, float] = {}
        for row, weight in zip(source_rows, weights, strict=True):
            values = [float(row[score_columns[label]]) for label in scored_labels]
            normalized_values = _normalize_score_values(values, normalized)
            for label, value in zip(scored_labels, normalized_values, strict=True):
                scores[label] = scores.get(label, 0.0) + weight * value
        if scores:
            if normalized == "raw" and source_weights is None:
                source = "class_score_mean"
            elif normalized == "raw":
                source = "class_score_weighted_mean"
            elif source_weights is not None:
                source = f"class_score_{normalized}_weighted_mean"
            else:
                source = f"class_score_{normalized}_mean"
            if mode == "score_tiebreak_first_source":
                source = f"{source}_tiebreak_first_source"
            tie_order = {label: index for index, label in enumerate(tie_break_labels)}
            return (
                sorted(
                    class_labels,
                    key=lambda label: (
                        -scores.get(label, float("-inf")),
                        tie_order.get(label, len(tie_order)),
                        label,
                    ),
                ),
                source,
            )

    if mode in {"mean_score", "score_tiebreak_first_source"}:
        return None

    if mode in {"auto", "mean_rank", "borda"}:
        rank_columns = _class_value_columns(source_rows, CLASS_RANK_PATTERNS)
    else:
        rank_columns = {}
    if rank_columns:
        weights = _normalized_source_weights(source_weights, len(source_rows))
        borda_scores: dict[int, float] = {}
        for label in class_labels:
            column = rank_columns.get(label)
            if column is None:
                continue
            values = [weight * float(row[column]) for row, weight in zip(source_rows, weights, strict=True)]
            borda_scores[label] = -sum(values)
        if borda_scores:
            rank_source = "class_rank_mean" if mode == "mean_rank" else "class_rank_borda"
            return sorted(class_labels, key=lambda label: (-borda_scores.get(label, float("-inf")), label)), rank_source

    return None


def _first_source_tie_break_labels(source_rows: Sequence[dict[str, str]], class_labels: Sequence[int]) -> list[int]:
    if not source_rows:
        return list(class_labels)
    first_prediction = _label_from_row(
        source_rows[0],
        label_column="predicted_label",
        stimulus_column="predicted_stimulus",
        field="predicted_label",
    )
    ranked = _rank_labels_by_scores(
        [source_rows[0]],
        class_labels=class_labels,
        aggregation_mode="mean_rank",
    )
    if ranked is not None:
        return ranked[0]
    ranked = _rank_labels_by_scores(
        [source_rows[0]],
        class_labels=class_labels,
        aggregation_mode="auto",
        tie_break_labels=[first_prediction, *[label for label in class_labels if label != first_prediction]],
    )
    if ranked is not None:
        return ranked[0]
    return [first_prediction, *[label for label in class_labels if label != first_prediction]]


def _sem(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) <= 1:
        return 0.0 if finite else math.nan
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance) / math.sqrt(len(finite))


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _balanced_accuracy(rows: Sequence[dict[str, object]]) -> float:
    by_label: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        by_label[_to_int(row["true_label"], field="true_label")].append(bool(row["correct"]))
    if not by_label:
        return math.nan
    recalls = [sum(values) / len(values) for values in by_label.values() if values]
    return sum(recalls) / len(recalls) if recalls else math.nan


def _format_percent(value: float) -> str:
    return "" if not math.isfinite(value) else f"{100.0 * value:.6f}"


def _prediction_row(
    *,
    ensemble_name: str,
    source_names: Sequence[str],
    source_weights: Sequence[float] | None,
    source_rows: Sequence[dict[str, str]],
    key_columns: Sequence[str],
    key: tuple[str, ...],
    class_labels: Sequence[int],
    score_normalization: str,
    aggregation_mode: str,
) -> dict[str, object]:
    aggregation_mode = _normalize_artifact_aggregation_mode(aggregation_mode)
    reference = source_rows[0]
    true_label = _label_from_row(reference, label_column="true_label", stimulus_column="true_stimulus", field="true_label")
    source_predictions = [
        _label_from_row(row, label_column="predicted_label", stimulus_column="predicted_stimulus", field="predicted_label")
        for row in source_rows
    ]
    if aggregation_mode == "hard_vote":
        ranked_labels = _rank_labels_by_hard_votes(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
        )
        rank_source = "hard_vote"
        true_rank = float(ranked_labels.index(true_label) + 1)
    elif len(source_rows) == 1:
        ranked = _rank_labels_by_scores(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
            score_normalization=score_normalization,
            aggregation_mode=aggregation_mode,
            tie_break_labels=_first_source_tie_break_labels(source_rows, class_labels),
        )
        if ranked is not None:
            ranked_labels, rank_source = ranked
            true_rank = float(ranked_labels.index(true_label) + 1)
        else:
            ranked_labels = [source_predictions[0], *[label for label in class_labels if label != source_predictions[0]]]
            source_true_rank = _source_rank(reference)
            true_rank = source_true_rank if source_true_rank is not None else float(ranked_labels.index(true_label) + 1)
            rank_source = "source_true_label_rank" if source_true_rank is not None else "hard_vote"
    else:
        ranked = _rank_labels_by_scores(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
            score_normalization=score_normalization,
            aggregation_mode=aggregation_mode,
            tie_break_labels=_first_source_tie_break_labels(source_rows, class_labels),
        )
        if ranked is None:
            ranked_labels = _rank_labels_by_hard_votes(
                source_rows,
                class_labels=class_labels,
                source_weights=source_weights,
            )
            rank_source = "hard_vote"
        else:
            ranked_labels, rank_source = ranked
        true_rank = float(ranked_labels.index(true_label) + 1)
    predicted_label = int(ranked_labels[0])
    row: dict[str, object] = {
        "artifact_ensemble": ensemble_name,
        "artifact_ensemble_sources": ";".join(source_names),
        "artifact_ensemble_source_count": len(source_names),
        "artifact_ensemble_requested_aggregation_mode": aggregation_mode,
        "artifact_ensemble_source_weights": (
            ""
            if source_weights is None
            else ";".join(
                f"{source_name}:{weight:.6g}"
                for source_name, weight in zip(source_names, _normalized_source_weights(source_weights, len(source_names)), strict=True)
            )
        ),
        "artifact_ensemble_score_normalization": _normalize_artifact_score_normalization(score_normalization),
        "artifact_ensemble_mode": (
            "hard_vote_tiebreak_first_source"
            if rank_source in {"hard_vote", "source_true_label_rank"}
            else rank_source
        ),
        "artifact_ensemble_rank_source": rank_source,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "true_stimulus": true_label + 1,
        "predicted_stimulus": predicted_label + 1,
        "correct": predicted_label == true_label,
        "true_label_rank": true_rank,
        "top2_correct": true_rank <= 2,
        "top3_correct": true_rank <= 3,
        "source_predicted_labels": ";".join(str(value) for value in source_predictions),
        "vote_ranked_labels": ";".join(str(value) for value in ranked_labels),
    }
    for column, value in zip(key_columns, key, strict=True):
        row[column] = value
    for optional in ("test_participant", "test_trial_index", "trial", "test_trial_number", "outer_fold"):
        if optional in reference and optional not in row:
            row[optional] = reference.get(optional, "")
    return row


def _outer_rows(ensemble_name: str, prediction_rows: Sequence[dict[str, object]], *, n_classes: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_participant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_participant[str(row.get("test_participant", ""))].append(row)
    for participant, participant_rows in sorted(by_participant.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        n_test = len(participant_rows)
        accuracy = sum(bool(row["correct"]) for row in participant_rows) / n_test if n_test else math.nan
        top2 = sum(bool(row["top2_correct"]) for row in participant_rows) / n_test if n_test else math.nan
        top3 = sum(bool(row["top3_correct"]) for row in participant_rows) / n_test if n_test else math.nan
        ranks = [_to_float(row["true_label_rank"]) for row in participant_rows]
        rows.append(
            {
                "artifact_ensemble": ensemble_name,
                "test_participant": participant,
                "accuracy": accuracy,
                "balanced_accuracy": _balanced_accuracy(participant_rows),
                "chance_accuracy": 1.0 / n_classes if n_classes else math.nan,
                "top2_accuracy": top2,
                "top3_accuracy": top3,
                "mean_true_label_rank": _mean(ranks),
                "n_test": n_test,
            }
        )
    return rows


def _participant_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def _metric_mean(rows: Sequence[dict[str, object]], metric: str) -> float:
    return _mean([_to_float(row[metric]) for row in rows])


def _counts_text(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ";".join(f"{value}:{counts[value]}" for value in sorted(counts, key=_participant_sort_key))


def _group_summary(
    ensemble_name: str,
    source_names: Sequence[str],
    outer_rows: Sequence[dict[str, object]],
    *,
    n_classes: int,
    source_weights: Sequence[float] | None = None,
    score_normalization: str = "raw",
    aggregation_mode: str = "auto",
) -> dict[str, object]:
    accuracies = [_to_float(row["accuracy"]) for row in outer_rows]
    balanced = [_to_float(row["balanced_accuracy"]) for row in outer_rows]
    top2 = [_to_float(row["top2_accuracy"]) for row in outer_rows]
    top3 = [_to_float(row["top3_accuracy"]) for row in outer_rows]
    ranks = [_to_float(row["mean_true_label_rank"]) for row in outer_rows]
    chance = 1.0 / n_classes if n_classes else math.nan
    return {
        "artifact_ensemble": ensemble_name,
        "artifact_ensemble_sources": ";".join(source_names),
        "artifact_ensemble_source_count": len(source_names),
        "artifact_ensemble_requested_aggregation_mode": _normalize_artifact_aggregation_mode(aggregation_mode),
        "artifact_ensemble_source_weights": (
            ""
            if source_weights is None
            else ";".join(
                f"{source_name}:{weight:.6g}"
                for source_name, weight in zip(source_names, _normalized_source_weights(source_weights, len(source_names)), strict=True)
            )
        ),
        "artifact_ensemble_score_normalization": _normalize_artifact_score_normalization(score_normalization),
        "n_outer_folds": len(outer_rows),
        "n_classes": n_classes,
        "chance_accuracy": chance,
        "accuracy_mean": _mean(accuracies),
        "accuracy_sem": _sem(accuracies),
        "balanced_accuracy_mean": _mean(balanced),
        "balanced_accuracy_sem": _sem(balanced),
        "top2_accuracy_mean": _mean(top2),
        "top2_accuracy_sem": _sem(top2),
        "top3_accuracy_mean": _mean(top3),
        "top3_accuracy_sem": _sem(top3),
        "mean_true_label_rank_mean": _mean(ranks),
        "mean_true_label_rank_sem": _sem(ranks),
        "accuracy_percent_mean": _format_percent(_mean(accuracies)),
        "balanced_percent_mean": _format_percent(_mean(balanced)),
        "top2_percent_mean": _format_percent(_mean(top2)),
        "top3_percent_mean": _format_percent(_mean(top3)),
        "participants_above_chance": sum(value > chance for value in balanced if math.isfinite(value) and math.isfinite(chance)),
    }


def _nested_subject_selector(
    *,
    selector_name: str,
    ensemble_order: Sequence[str],
    ensemble_sources: dict[str, Sequence[str]],
    prediction_rows_by_ensemble: dict[str, list[dict]],
    outer_rows_by_ensemble: dict[str, list[dict]],
    n_classes: int,
    score_normalization: str,
    aggregation_mode: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, object]]:
    """Select an artifact ensemble recipe for each subject using other subjects only."""

    outer_by_ensemble_participant = {
        ensemble: {str(row.get("test_participant", "")): row for row in rows}
        for ensemble, rows in outer_rows_by_ensemble.items()
    }
    prediction_by_ensemble_participant: dict[str, dict[str, list[dict]]] = {}
    for ensemble, rows in prediction_rows_by_ensemble.items():
        by_participant: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_participant[str(row.get("test_participant", ""))].append(row)
        prediction_by_ensemble_participant[ensemble] = by_participant

    participants = sorted(
        set().union(*(set(rows) for rows in outer_by_ensemble_participant.values())),
        key=_participant_sort_key,
    )
    if not participants:
        raise ValueError("Nested subject selector requires test_participant values.")

    selected_predictions: list[dict] = []
    selection_rows: list[dict] = []
    for participant in participants:
        candidates: list[tuple[float, int, str, list[dict]]] = []
        for ensemble_index, ensemble in enumerate(ensemble_order):
            train_outer_rows = [
                row
                for other_participant, row in outer_by_ensemble_participant[ensemble].items()
                if other_participant != participant
            ]
            if not train_outer_rows:
                raise ValueError(f"Cannot select an artifact ensemble for participant {participant}; no source subjects remain.")
            balanced = _metric_mean(train_outer_rows, "balanced_accuracy")
            candidates.append((balanced, -ensemble_index, ensemble, train_outer_rows))

        selected_balanced, _, selected_ensemble, train_outer_rows = max(candidates)
        selected_sources = ensemble_sources[selected_ensemble]
        participant_predictions = prediction_by_ensemble_participant[selected_ensemble].get(participant)
        if not participant_predictions:
            raise ValueError(f"Selected ensemble {selected_ensemble!r} has no predictions for participant {participant}.")

        selection_rows.append(
            {
                "test_participant": participant,
                "artifact_ensemble": selector_name,
                "selected_artifact_ensemble": selected_ensemble,
                "selected_artifact_ensemble_sources": ";".join(selected_sources),
                "selection_metric": "other_subjects_balanced_accuracy",
                "selection_metric_value": selected_balanced,
                "selection_accuracy": _metric_mean(train_outer_rows, "accuracy"),
                "selection_top2_accuracy": _metric_mean(train_outer_rows, "top2_accuracy"),
                "selection_top3_accuracy": _metric_mean(train_outer_rows, "top3_accuracy"),
                "selection_n_subjects": len(train_outer_rows),
                "candidate_artifact_ensembles": ";".join(ensemble_order),
            }
        )
        for row in participant_predictions:
            selected_row = dict(row)
            selected_row["artifact_ensemble"] = selector_name
            selected_row["artifact_ensemble_recipe_selection"] = "leave_subject_out"
            selected_row["selected_artifact_ensemble"] = selected_ensemble
            selected_row["selected_artifact_ensemble_sources"] = ";".join(selected_sources)
            selected_row["selection_metric"] = "other_subjects_balanced_accuracy"
            selected_row["selection_metric_value"] = selected_balanced
            selected_predictions.append(selected_row)

    outer_rows = _outer_rows(selector_name, selected_predictions, n_classes=n_classes)
    summary = _group_summary(
        selector_name,
        ensemble_order,
        outer_rows,
        n_classes=n_classes,
        score_normalization=score_normalization,
        aggregation_mode=aggregation_mode,
    )
    summary["artifact_ensemble_recipe_selection"] = "leave_subject_out"
    summary["selection_metric"] = "other_subjects_balanced_accuracy"
    summary["selected_artifact_ensemble_counts"] = _counts_text(
        str(row["selected_artifact_ensemble"]) for row in selection_rows
    )
    return selected_predictions, outer_rows, selection_rows, summary


def _prediction_key_sort_key(values: tuple[str, ...]) -> tuple[tuple[int, int | str], ...]:
    return tuple((0, int(value)) if value.isdigit() else (1, value) for value in values)


def ensemble_prediction_sources(
    sources: Sequence[PredictionSource],
    ensembles: Sequence[tuple[str, Sequence[str]] | tuple[str, Sequence[str], Sequence[float] | None]],
    *,
    key_columns: Sequence[str] = DEFAULT_KEY_COLUMNS,
    nested_selector_name: str | None = None,
    score_normalization: str = "raw",
    aggregation_mode: str = "auto",
) -> dict[str, list[dict]]:
    """Build hard-vote artifact ensembles from already completed prediction CSVs."""

    score_normalization = _normalize_artifact_score_normalization(score_normalization)
    aggregation_mode = _normalize_artifact_aggregation_mode(aggregation_mode)
    source_by_name = {source.name: source for source in sources}
    if len(source_by_name) != len(sources):
        raise ValueError("Source names must be unique.")
    class_labels = sorted({label for source in sources for label in _all_labels(source.rows)})
    if not class_labels:
        raise ValueError("Could not infer class labels from prediction rows.")

    indexed_sources = {source.name: _index_rows(source, key_columns) for source in sources}
    all_predictions: list[dict] = []
    all_outer: list[dict] = []
    all_summary: list[dict] = []
    prediction_rows_by_ensemble: dict[str, list[dict]] = {}
    outer_rows_by_ensemble: dict[str, list[dict]] = {}
    ensemble_sources: dict[str, Sequence[str]] = {}
    for ensemble_entry in ensembles:
        ensemble_name, source_names, source_weights = _normalize_ensemble_entry(ensemble_entry)
        missing_sources = [name for name in source_names if name not in source_by_name]
        if missing_sources:
            raise ValueError(f"Unknown ensemble source(s) for {ensemble_name}: {', '.join(missing_sources)}")
        if ensemble_name in ensemble_sources:
            raise ValueError(f"Artifact ensemble names must be unique; got duplicate {ensemble_name!r}.")
        ensemble_sources[ensemble_name] = tuple(source_names)
        reference_keys = set(indexed_sources[source_names[0]])
        for source_name in source_names[1:]:
            keys = set(indexed_sources[source_name])
            if keys != reference_keys:
                missing = sorted(reference_keys - keys)[:5]
                extra = sorted(keys - reference_keys)[:5]
                raise ValueError(
                    f"Prediction keys do not match for ensemble {ensemble_name!r} source {source_name!r}; "
                    f"missing examples={missing}, extra examples={extra}."
                )
        prediction_rows = [
            _prediction_row(
                ensemble_name=ensemble_name,
                source_names=source_names,
                source_weights=source_weights,
                source_rows=[indexed_sources[source_name][key] for source_name in source_names],
                key_columns=key_columns,
                key=key,
                class_labels=class_labels,
                score_normalization=score_normalization,
                aggregation_mode=aggregation_mode,
            )
            for key in sorted(reference_keys, key=_prediction_key_sort_key)
        ]
        outer_rows = _outer_rows(ensemble_name, prediction_rows, n_classes=len(class_labels))
        summary = _group_summary(
            ensemble_name,
            source_names,
            outer_rows,
            n_classes=len(class_labels),
            source_weights=source_weights,
            score_normalization=score_normalization,
            aggregation_mode=aggregation_mode,
        )
        prediction_rows_by_ensemble[ensemble_name] = prediction_rows
        outer_rows_by_ensemble[ensemble_name] = outer_rows
        all_predictions.extend(prediction_rows)
        all_outer.extend(outer_rows)
        all_summary.append(summary)
    artifacts: dict[str, list[dict]] = {"predictions": all_predictions, "outer": all_outer, "group_summary": all_summary}
    if nested_selector_name:
        nested_predictions, nested_outer, nested_selection, nested_summary = _nested_subject_selector(
            selector_name=nested_selector_name,
            ensemble_order=[ensemble_entry[0] for ensemble_entry in ensembles],
            ensemble_sources=ensemble_sources,
            prediction_rows_by_ensemble=prediction_rows_by_ensemble,
            outer_rows_by_ensemble=outer_rows_by_ensemble,
            n_classes=len(class_labels),
            score_normalization=score_normalization,
            aggregation_mode=aggregation_mode,
        )
        artifacts["predictions"].extend(nested_predictions)
        artifacts["outer"].extend(nested_outer)
        artifacts["group_summary"].append(nested_summary)
        artifacts["nested_selection"] = nested_selection
    return artifacts


def _normalize_ensemble_entry(
    ensemble_entry: tuple[str, Sequence[str]] | tuple[str, Sequence[str], Sequence[float] | None],
) -> tuple[str, tuple[str, ...], tuple[float, ...] | None]:
    if len(ensemble_entry) == 2:
        ensemble_name, source_names = ensemble_entry
        source_weights = None
    elif len(ensemble_entry) == 3:
        ensemble_name, source_names, source_weights = ensemble_entry
    else:
        raise ValueError("Ensemble entries must contain name/sources or name/sources/weights.")
    source_names = tuple(source_names)
    if source_weights is not None:
        source_weights = _normalized_source_weights(source_weights, len(source_names))
    return str(ensemble_name), source_names, source_weights


def write_markdown_summary(path: Path, summary_rows: Sequence[dict]) -> None:
    lines = [
        "# Artifact Prediction Ensembles",
        "",
        "| ensemble | mode | sources | folds | balanced | accuracy | top-2 | top-3 | mean rank |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {mode} | {sources} | {folds} | {balanced:.2f}% | {accuracy:.2f}% | {top2:.2f}% | {top3:.2f}% | {rank:.3f} |".format(
                name=row["artifact_ensemble"],
                mode=row.get("artifact_ensemble_requested_aggregation_mode", row.get("artifact_ensemble_score_normalization", "")),
                sources=str(row["artifact_ensemble_sources"]).replace(";", ", "),
                folds=row["n_outer_folds"],
                balanced=100.0 * float(row["balanced_accuracy_mean"]),
                accuracy=100.0 * float(row["accuracy_mean"]),
                top2=100.0 * float(row["top2_accuracy_mean"]),
                top3=100.0 * float(row["top3_accuracy_mean"]),
                rank=float(row["mean_true_label_rank_mean"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="Named prediction artifact source as name=path.")
    parser.add_argument(
        "--ensemble",
        action="append",
        required=True,
        help="Named ensemble as name=source_a,source_b or weighted name=source_a:0.8,source_b:0.2.",
    )
    parser.add_argument("--key-column", action="append", dest="key_columns", help="Prediction-row key column. Defaults to test_participant, test_trial_index, true_label.")
    parser.add_argument(
        "--score-normalization",
        choices=ARTIFACT_SCORE_NORMALIZATION_CHOICES,
        default="raw",
        help="Per-source trial-level score normalization before averaging class scores.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=ARTIFACT_AGGREGATION_MODE_CHOICES,
        default="auto",
        help="How to combine source predictions: score mean, rank/Borda, hard vote, or automatic score-then-rank fallback.",
    )
    parser.add_argument(
        "--nested-selector-name",
        help="Optional leakage-safe leave-subject-out artifact recipe selector row to add to the outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="artifact_ensemble")
    args = parser.parse_args(argv)

    sources = [load_prediction_source(spec) for spec in args.source]
    ensembles = [parse_weighted_ensemble_spec(spec) for spec in args.ensemble]
    artifacts = ensemble_prediction_sources(
        sources,
        ensembles,
        key_columns=tuple(args.key_columns or DEFAULT_KEY_COLUMNS),
        nested_selector_name=args.nested_selector_name,
        score_normalization=args.score_normalization,
        aggregation_mode=args.aggregation_mode,
    )
    write_csv_rows(args.output_dir / f"{args.output_stem}_predictions.csv", artifacts["predictions"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_outer.csv", artifacts["outer"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_group_summary.csv", artifacts["group_summary"])
    if "nested_selection" in artifacts:
        write_csv_rows(args.output_dir / f"{args.output_stem}_nested_selection.csv", artifacts["nested_selection"])
    write_markdown_summary(args.output_dir / f"{args.output_stem}_comparison.md", artifacts["group_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
