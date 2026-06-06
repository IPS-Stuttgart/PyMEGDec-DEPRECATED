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
    "log_score_mean",
    "mean_rank",
    "borda",
    "score_tiebreak_first_source",
    "balanced_assignment",
)
ARTIFACT_NESTED_SELECTION_METRIC_CHOICES = (
    "balanced_accuracy",
    "balanced_accuracy_lcb",
    "balanced_top2_top3_rank",
    "balanced_top2_top3_rank_lcb",
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


def _display_label_map(class_labels: Sequence[int]) -> dict[int, int]:
    values = [int(label) for label in class_labels]
    if values and min(values) == 0 and max(values) == len(values) - 1:
        return {label: label + 1 for label in values}
    return {label: label for label in values}


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


def _safe_log_score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        return math.log(1e-300)
    return math.log(value)


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
        "log_score": "log_score_mean",
        "mean_log_score": "log_score_mean",
        "geometric_mean": "log_score_mean",
        "geometric_score_mean": "log_score_mean",
        "product_score": "log_score_mean",
        "quota": "balanced_assignment",
        "balanced": "balanced_assignment",
        "uniform_balanced_assignment": "balanced_assignment",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ARTIFACT_AGGREGATION_MODE_CHOICES:
        raise ValueError(
            "Artifact aggregation mode must be one of "
            f"{', '.join(ARTIFACT_AGGREGATION_MODE_CHOICES)}."
        )
    return normalized


def _normalize_artifact_nested_selection_metric(selection_metric: str) -> str:
    normalized = str(selection_metric).strip().lower().replace("-", "_")
    aliases = {
        "balanced": "balanced_accuracy",
        "balanced_lcb": "balanced_accuracy_lcb",
        "balanced_rank": "balanced_top2_top3_rank",
        "balanced_rank_lcb": "balanced_top2_top3_rank_lcb",
        "rank_lcb": "balanced_top2_top3_rank_lcb",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ARTIFACT_NESTED_SELECTION_METRIC_CHOICES:
        raise ValueError(
            "Artifact nested selection metric must be one of "
            f"{', '.join(ARTIFACT_NESTED_SELECTION_METRIC_CHOICES)}."
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


def _aggregate_class_scores(
    source_rows: Sequence[dict[str, str]],
    *,
    class_labels: Sequence[int],
    source_weights: Sequence[float] | None = None,
    score_normalization: str = "raw",
    use_log_scores: bool = False,
) -> tuple[dict[int, float], str] | None:
    score_columns = _class_value_columns(source_rows, CLASS_SCORE_PATTERNS)
    if not score_columns:
        return None
    normalized = _normalize_artifact_score_normalization(score_normalization)
    weights = _normalized_source_weights(source_weights, len(source_rows))
    scored_labels = [label for label in class_labels if label in score_columns]
    if not scored_labels:
        return None
    scores: dict[int, float] = {}
    for row, weight in zip(source_rows, weights, strict=True):
        values = [float(row[score_columns[label]]) for label in scored_labels]
        normalized_values = _normalize_score_values(values, normalized)
        for label, value in zip(scored_labels, normalized_values, strict=True):
            contribution = _safe_log_score(value) if use_log_scores else value
            scores[label] = scores.get(label, 0.0) + weight * contribution
    if use_log_scores and normalized == "raw" and source_weights is None:
        source = "class_score_log_mean"
    elif use_log_scores and normalized == "raw":
        source = "class_score_log_weighted_mean"
    elif use_log_scores and source_weights is not None:
        source = f"class_score_{normalized}_log_weighted_mean"
    elif use_log_scores:
        source = f"class_score_{normalized}_log_mean"
    elif normalized == "raw" and source_weights is None:
        source = "class_score_mean"
    elif normalized == "raw":
        source = "class_score_weighted_mean"
    elif source_weights is not None:
        source = f"class_score_{normalized}_weighted_mean"
    else:
        source = f"class_score_{normalized}_mean"
    return scores, source


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
    score_modes = {"auto", "mean_score", "log_score_mean", "score_tiebreak_first_source", "balanced_assignment"}
    aggregated = None
    if mode in score_modes:
        aggregated = _aggregate_class_scores(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
            score_normalization=score_normalization,
            use_log_scores=mode == "log_score_mean",
        )
    if aggregated is not None:
        scores, source = aggregated
        if mode == "score_tiebreak_first_source":
            source = f"{source}_tiebreak_first_source"
        if mode == "balanced_assignment":
            source = f"{source}_balanced_assignment_candidate"
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

    if mode in {"mean_score", "log_score_mean", "score_tiebreak_first_source"}:
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
    display_labels = _display_label_map(class_labels)
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
        "true_stimulus": display_labels.get(true_label, true_label),
        "predicted_stimulus": display_labels.get(predicted_label, predicted_label),
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
    aggregated = _aggregate_class_scores(
        source_rows,
        class_labels=class_labels,
        source_weights=source_weights,
        score_normalization=score_normalization,
        use_log_scores=aggregation_mode == "log_score_mean",
    )
    if aggregated is not None:
        aggregate_scores, _score_source = aggregated
        for label in class_labels:
            if label in aggregate_scores:
                row[f"artifact_score_class_{label}"] = aggregate_scores[label]
    return row


def _uniform_class_quotas(n_items: int, n_classes: int) -> list[int]:
    if n_classes <= 0:
        return []
    quotas = [int(n_items) // int(n_classes) for _ in range(int(n_classes))]
    for index in range(int(n_items) - sum(quotas)):
        quotas[index] += 1
    return quotas


def _balanced_assignment_indices(score_matrix: Sequence[Sequence[float]], quotas: Sequence[int]) -> tuple[list[int], float]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy is normally installed through sklearn.
        raise RuntimeError("artifact aggregation balanced_assignment requires scipy.") from exc

    scores = [list(map(float, row)) for row in score_matrix]
    if not scores:
        return [], 0.0
    repeated_class_indices: list[int] = []
    for class_index, quota in enumerate(quotas):
        repeated_class_indices.extend([class_index] * int(quota))
    if len(repeated_class_indices) != len(scores):
        raise ValueError("Balanced-assignment quotas must sum to the number of rows.")
    cost = [[-row[class_index] for class_index in repeated_class_indices] for row in scores]
    row_indices, assignment_columns = linear_sum_assignment(cost)
    predicted = [0 for _ in scores]
    for row_index, assignment_column in zip(row_indices, assignment_columns, strict=True):
        predicted[int(row_index)] = int(repeated_class_indices[int(assignment_column)])
    argmax_score = sum(max(row) for row in scores)
    assignment_score = sum(scores[row_index][class_index] for row_index, class_index in enumerate(predicted))
    return predicted, float(assignment_score - argmax_score)


def _apply_balanced_assignment_rows(prediction_rows: Sequence[dict[str, object]], class_labels: Sequence[int]) -> list[dict[str, object]]:
    rows = [dict(row) for row in prediction_rows]
    label_list = [int(label) for label in class_labels]
    display_labels = _display_label_map(label_list)
    by_participant: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_participant[str(row.get("test_participant", ""))].append(row_index)
    for indices in by_participant.values():
        score_columns = [f"artifact_score_class_{label}" for label in label_list]
        missing = [column for column in score_columns if any(str(rows[index].get(column, "")).strip() == "" for index in indices)]
        if missing:
            raise ValueError(
                "balanced_assignment requires class score/probability columns for every class; "
                f"missing examples={missing[:5]}."
            )
        score_matrix = [[_to_float(rows[index][column]) for column in score_columns] for index in indices]
        quotas = _uniform_class_quotas(len(indices), len(label_list))
        assigned_indices, objective_delta = _balanced_assignment_indices(score_matrix, quotas)
        quota_text = ";".join(f"{label}:{quota}" for label, quota in zip(label_list, quotas, strict=True))
        for row_index, assigned_index in zip(indices, assigned_indices, strict=True):
            row = rows[row_index]
            predicted_label = label_list[int(assigned_index)]
            true_label = _to_int(row["true_label"], field="true_label")
            row["predicted_label"] = predicted_label
            row["predicted_stimulus"] = display_labels.get(predicted_label, predicted_label)
            row["correct"] = predicted_label == true_label
            row["artifact_ensemble_mode"] = "class_score_balanced_assignment"
            row["artifact_ensemble_balanced_assignment_quota_counts"] = quota_text
            row["artifact_ensemble_balanced_assignment_objective_delta"] = objective_delta
    return rows


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


def _nested_selection_metric_label(selection_metric: str) -> str:
    return f"other_subjects_{_normalize_artifact_nested_selection_metric(selection_metric)}"


def _outer_metric_values(rows: Sequence[dict[str, object]], metric: str) -> list[float]:
    return [_to_float(row[metric]) for row in rows]


def _artifact_recipe_rank_score(row: dict[str, object], *, n_classes: int) -> float:
    """Return a source-only recipe score that rewards top-k/rank signal.

    The balanced-accuracy term remains dominant. Top-2/top-3 and mean-rank
    terms are deliberately small, chance-centered nudges intended for artifact
    recipe selection when several recipes tie on balanced accuracy across the
    other subjects.
    """

    if n_classes <= 0:
        raise ValueError("Nested artifact selection requires at least one class.")
    balanced = _to_float(row["balanced_accuracy"])
    top2 = _to_float(row["top2_accuracy"])
    top3 = _to_float(row["top3_accuracy"])
    mean_rank = _to_float(row["mean_true_label_rank"])
    top2_chance = min(2.0 / n_classes, 1.0)
    top3_chance = min(3.0 / n_classes, 1.0)
    chance_mean_rank = 0.5 * (n_classes + 1.0)
    rank_scale = max(chance_mean_rank - 1.0, 1.0)
    rank_gain = (chance_mean_rank - mean_rank) / rank_scale
    return balanced + 0.25 * (top2 - top2_chance) + 0.125 * (top3 - top3_chance) + 0.10 * rank_gain


def _nested_selection_metric_value(rows: Sequence[dict[str, object]], *, selection_metric: str, n_classes: int) -> float:
    normalized = _normalize_artifact_nested_selection_metric(selection_metric)
    if normalized == "balanced_accuracy":
        return _metric_mean(rows, "balanced_accuracy")
    if normalized == "balanced_accuracy_lcb":
        values = _outer_metric_values(rows, "balanced_accuracy")
        return _mean(values) - _sem(values)

    values = [_artifact_recipe_rank_score(row, n_classes=n_classes) for row in rows]
    score = _mean(values)
    if normalized.endswith("_lcb"):
        score -= _sem(values)
    return score


def _counts_text(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ";".join(f"{value}:{counts[value]}" for value in sorted(counts, key=_participant_sort_key))


@dataclass(frozen=True)
class WeightCandidate:
    candidate_index: int
    weights: tuple[float, ...]
    weight_text: str
    uniform_distance: float
    predictions_by_participant: dict[str, list[dict]]
    outer_by_participant: dict[str, dict]


def _source_weight_text(source_names: Sequence[str], weights: Sequence[float]) -> str:
    normalized = _normalized_source_weights(weights, len(source_names))
    return ";".join(
        f"{source_name}:{weight:.6g}"
        for source_name, weight in zip(source_names, normalized, strict=True)
    )


def _simplex_weight_grid(n_sources: int, step: float) -> list[tuple[float, ...]]:
    """Return non-negative source-weight vectors on a simplex grid.

    The grid is intentionally small and deterministic; it is meant for
    leakage-safe artifact-level selection, not for fitting another model to the
    held-out subject.  ``step=0.25`` gives 15 candidates for three sources.
    """

    if n_sources <= 0:
        raise ValueError("At least one source is required for a weight grid.")
    if step <= 0.0 or step > 1.0 or not math.isfinite(step):
        raise ValueError("Weight-grid step must be finite and in the interval (0, 1].")
    denominator_float = 1.0 / step
    denominator = int(round(denominator_float))
    if denominator <= 0 or not math.isclose(denominator_float, denominator, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Weight-grid step must evenly divide 1.0, e.g. 1, 0.5, 0.25, or 0.1.")

    weights: list[tuple[float, ...]] = []

    def visit(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(tuple(value / denominator for value in (*prefix, remaining)))
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots - 1)

    visit([], denominator, n_sources)
    return weights


def _weight_uniform_distance(weights: Sequence[float]) -> float:
    if not weights:
        return math.inf
    uniform = 1.0 / len(weights)
    return sum(abs(weight - uniform) for weight in weights)


def _weighted_ensemble_predictions(
    *,
    ensemble_name: str,
    source_names: Sequence[str],
    source_weights: Sequence[float],
    indexed_sources: dict[str, dict[tuple[str, ...], dict[str, str]]],
    key_columns: Sequence[str],
    class_labels: Sequence[int],
    score_normalization: str,
    aggregation_mode: str,
) -> list[dict]:
    reference_keys = set(indexed_sources[source_names[0]])
    return [
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


def _nested_source_weight_selector(
    *,
    selector_name: str,
    selector_ensemble: str,
    ensemble_sources: dict[str, Sequence[str]],
    indexed_sources: dict[str, dict[tuple[str, ...], dict[str, str]]],
    key_columns: Sequence[str],
    class_labels: Sequence[int],
    n_classes: int,
    score_normalization: str,
    aggregation_mode: str,
    grid_step: float,
    nested_weight_selection_metric: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, object]]:
    """Choose source weights per held-out subject using other subjects only."""

    nested_weight_selection_metric = _normalize_artifact_nested_selection_metric(nested_weight_selection_metric)
    if selector_ensemble not in ensemble_sources:
        raise ValueError(f"Unknown nested weight-selector ensemble {selector_ensemble!r}.")
    source_names = tuple(ensemble_sources[selector_ensemble])
    if len(source_names) < 2:
        raise ValueError("Nested weight selection requires an ensemble with at least two sources.")

    candidates: list[WeightCandidate] = []
    for candidate_index, weights in enumerate(_simplex_weight_grid(len(source_names), grid_step)):
        prediction_rows = _weighted_ensemble_predictions(
            ensemble_name=f"{selector_name}__candidate_{candidate_index}",
            source_names=source_names,
            source_weights=weights,
            indexed_sources=indexed_sources,
            key_columns=key_columns,
            class_labels=class_labels,
            score_normalization=score_normalization,
            aggregation_mode=aggregation_mode,
        )
        outer_rows = _outer_rows(f"{selector_name}__candidate_{candidate_index}", prediction_rows, n_classes=n_classes)
        by_participant: dict[str, list[dict]] = defaultdict(list)
        for row in prediction_rows:
            by_participant[str(row.get("test_participant", ""))].append(row)
        candidates.append(
            WeightCandidate(
                candidate_index=candidate_index,
                weights=weights,
                weight_text=_source_weight_text(source_names, weights),
                uniform_distance=_weight_uniform_distance(weights),
                predictions_by_participant=by_participant,
                outer_by_participant={str(row.get("test_participant", "")): row for row in outer_rows},
            )
        )

    participants = sorted(
        set().union(*(set(candidate.outer_by_participant) for candidate in candidates)),
        key=_participant_sort_key,
    )
    if not participants:
        raise ValueError("Nested source-weight selector requires test_participant values.")

    selected_predictions: list[dict] = []
    selection_rows: list[dict] = []
    for participant in participants:
        scored_candidates: list[tuple[float, float, float, int, WeightCandidate, list[dict]]] = []
        for candidate in candidates:
            train_outer_rows = [
                row
                for other_participant, row in candidate.outer_by_participant.items()
                if other_participant != participant
            ]
            if not train_outer_rows:
                raise ValueError(f"Cannot select artifact source weights for participant {participant}; no source subjects remain.")
            selection_score = _nested_selection_metric_value(
                train_outer_rows,
                selection_metric=nested_weight_selection_metric,
                n_classes=n_classes,
            )
            balanced = _metric_mean(train_outer_rows, "balanced_accuracy")
            scored_candidates.append(
                (
                    selection_score,
                    balanced,
                    -candidate.uniform_distance,
                    -candidate.candidate_index,
                    candidate,
                    train_outer_rows,
                )
            )

        selected_score, selected_balanced, _, _, selected_candidate, train_outer_rows = max(scored_candidates)
        participant_predictions = selected_candidate.predictions_by_participant.get(participant)
        if not participant_predictions:
            raise ValueError(f"Selected source-weight candidate has no predictions for participant {participant}.")

        weight_text = selected_candidate.weight_text
        selection_rows.append(
            {
                "test_participant": participant,
                "artifact_ensemble": selector_name,
                "selected_artifact_ensemble": selector_ensemble,
                "selected_artifact_ensemble_sources": ";".join(source_names),
                "selected_source_weights": weight_text,
                "selected_weight_grid_step": grid_step,
                "selection_metric": _nested_selection_metric_label(nested_weight_selection_metric),
                "selection_metric_name": nested_weight_selection_metric,
                "selection_metric_value": selected_score,
                "selection_balanced_accuracy": selected_balanced,
                "selection_accuracy": _metric_mean(train_outer_rows, "accuracy"),
                "selection_top2_accuracy": _metric_mean(train_outer_rows, "top2_accuracy"),
                "selection_top3_accuracy": _metric_mean(train_outer_rows, "top3_accuracy"),
                "selection_mean_true_label_rank": _metric_mean(train_outer_rows, "mean_true_label_rank"),
                "selection_n_subjects": len(train_outer_rows),
                "candidate_source_weight_count": len(candidates),
            }
        )
        for row in participant_predictions:
            selected_row = dict(row)
            selected_row["artifact_ensemble"] = selector_name
            selected_row["artifact_ensemble_weight_selection"] = "leave_subject_out_grid"
            selected_row["selected_artifact_ensemble"] = selector_ensemble
            selected_row["selected_artifact_ensemble_sources"] = ";".join(source_names)
            selected_row["selected_source_weights"] = weight_text
            selected_row["selected_weight_grid_step"] = grid_step
            selected_row["selection_metric"] = _nested_selection_metric_label(nested_weight_selection_metric)
            selected_row["selection_metric_name"] = nested_weight_selection_metric
            selected_row["selection_metric_value"] = selected_score
            selected_row["selection_balanced_accuracy"] = selected_balanced
            selected_predictions.append(selected_row)

    outer_rows = _outer_rows(selector_name, selected_predictions, n_classes=n_classes)
    summary = _group_summary(
        selector_name,
        source_names,
        outer_rows,
        n_classes=n_classes,
        score_normalization=score_normalization,
        aggregation_mode=aggregation_mode,
    )
    summary["artifact_ensemble_weight_selection"] = "leave_subject_out_grid"
    summary["selected_artifact_ensemble"] = selector_ensemble
    summary["selection_metric"] = _nested_selection_metric_label(nested_weight_selection_metric)
    summary["selection_metric_name"] = nested_weight_selection_metric
    summary["selected_source_weight_counts"] = _counts_text(str(row["selected_source_weights"]) for row in selection_rows)
    summary["candidate_source_weight_count"] = len(candidates)
    summary["selected_weight_grid_step"] = grid_step
    return selected_predictions, outer_rows, selection_rows, summary


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
    nested_selection_metric: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, object]]:
    """Select an artifact ensemble recipe for each subject using other subjects only."""

    nested_selection_metric = _normalize_artifact_nested_selection_metric(nested_selection_metric)
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
            selection_score = _nested_selection_metric_value(train_outer_rows, selection_metric=nested_selection_metric, n_classes=n_classes)
            candidates.append((selection_score, -ensemble_index, ensemble, train_outer_rows))

        selected_score, _, selected_ensemble, train_outer_rows = max(candidates)
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
                "selection_metric": _nested_selection_metric_label(nested_selection_metric),
                "selection_metric_name": nested_selection_metric,
                "selection_metric_value": selected_score,
                "selection_balanced_accuracy": _metric_mean(train_outer_rows, "balanced_accuracy"),
                "selection_accuracy": _metric_mean(train_outer_rows, "accuracy"),
                "selection_top2_accuracy": _metric_mean(train_outer_rows, "top2_accuracy"),
                "selection_top3_accuracy": _metric_mean(train_outer_rows, "top3_accuracy"),
                "selection_mean_true_label_rank": _metric_mean(train_outer_rows, "mean_true_label_rank"),
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
            selected_row["selection_metric"] = _nested_selection_metric_label(nested_selection_metric)
            selected_row["selection_metric_name"] = nested_selection_metric
            selected_row["selection_metric_value"] = selected_score
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
    summary["selection_metric"] = _nested_selection_metric_label(nested_selection_metric)
    summary["selection_metric_name"] = nested_selection_metric
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
    nested_weight_selector_name: str | None = None,
    nested_weight_selector_ensemble: str | None = None,
    nested_weight_grid_step: float = 0.25,
    score_normalization: str = "raw",
    aggregation_mode: str = "auto",
    nested_selection_metric: str = "balanced_accuracy",
    nested_weight_selection_metric: str = "balanced_accuracy",
) -> dict[str, list[dict]]:
    """Build leakage-safe artifact ensembles from completed prediction CSVs."""

    score_normalization = _normalize_artifact_score_normalization(score_normalization)
    aggregation_mode = _normalize_artifact_aggregation_mode(aggregation_mode)
    nested_selection_metric = _normalize_artifact_nested_selection_metric(nested_selection_metric)
    nested_weight_selection_metric = _normalize_artifact_nested_selection_metric(nested_weight_selection_metric)
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
        if aggregation_mode == "balanced_assignment":
            prediction_rows = _apply_balanced_assignment_rows(prediction_rows, class_labels)
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
            nested_selection_metric=nested_selection_metric,
        )
        artifacts["predictions"].extend(nested_predictions)
        artifacts["outer"].extend(nested_outer)
        artifacts["group_summary"].append(nested_summary)
        artifacts["nested_selection"] = nested_selection
    if nested_weight_selector_name:
        if nested_weight_selector_ensemble is None:
            multi_source_ensembles = [name for name in ensemble_sources if len(ensemble_sources[name]) > 1]
            if not multi_source_ensembles:
                raise ValueError("Nested weight selection requires at least one multi-source ensemble.")
            nested_weight_selector_ensemble = multi_source_ensembles[0]
        weight_predictions, weight_outer, weight_selection, weight_summary = _nested_source_weight_selector(
            selector_name=nested_weight_selector_name,
            selector_ensemble=nested_weight_selector_ensemble,
            ensemble_sources=ensemble_sources,
            indexed_sources=indexed_sources,
            key_columns=key_columns,
            class_labels=class_labels,
            n_classes=len(class_labels),
            score_normalization=score_normalization,
            aggregation_mode=aggregation_mode,
            grid_step=nested_weight_grid_step,
            nested_weight_selection_metric=nested_weight_selection_metric,
        )
        artifacts["predictions"].extend(weight_predictions)
        artifacts["outer"].extend(weight_outer)
        artifacts["group_summary"].append(weight_summary)
        artifacts["nested_weight_selection"] = weight_selection
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
        help="How to combine source predictions: score/log-score mean, rank/Borda, hard vote, balanced assignment, or automatic score-then-rank fallback.",
    )
    parser.add_argument(
        "--nested-selection-metric",
        choices=ARTIFACT_NESTED_SELECTION_METRIC_CHOICES,
        default="balanced_accuracy",
        help="Leave-subject-out metric for selecting artifact ensemble recipes and nested source-weight candidates.",
    )
    parser.add_argument(
        "--nested-weight-selection-metric",
        choices=ARTIFACT_NESTED_SELECTION_METRIC_CHOICES,
        default="balanced_accuracy",
        help="Leave-subject-out metric for selecting source-weight grid candidates.",
    )
    parser.add_argument(
        "--nested-selector-name",
        help="Optional leakage-safe leave-subject-out artifact recipe selector row to add to the outputs.",
    )
    parser.add_argument(
        "--nested-weight-selector-name",
        help="Optional leakage-safe leave-subject-out source-weight selector row to add to the outputs.",
    )
    parser.add_argument(
        "--nested-weight-selector-ensemble",
        help="Ensemble whose source weights should be grid-selected by --nested-weight-selector-name. Defaults to the first multi-source ensemble.",
    )
    parser.add_argument(
        "--nested-weight-grid-step",
        type=float,
        default=0.25,
        help="Simplex grid step for nested source-weight selection. Use values that divide 1.0, e.g. 0.5, 0.25, or 0.1.",
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
        nested_weight_selector_name=args.nested_weight_selector_name,
        nested_weight_selector_ensemble=args.nested_weight_selector_ensemble,
        nested_weight_grid_step=args.nested_weight_grid_step,
        score_normalization=args.score_normalization,
        aggregation_mode=args.aggregation_mode,
        nested_selection_metric=args.nested_selection_metric,
        nested_weight_selection_metric=args.nested_weight_selection_metric,
    )
    write_csv_rows(args.output_dir / f"{args.output_stem}_predictions.csv", artifacts["predictions"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_outer.csv", artifacts["outer"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_group_summary.csv", artifacts["group_summary"])
    if "nested_selection" in artifacts:
        write_csv_rows(args.output_dir / f"{args.output_stem}_nested_selection.csv", artifacts["nested_selection"])
    if "nested_weight_selection" in artifacts:
        write_csv_rows(args.output_dir / f"{args.output_stem}_nested_weight_selection.csv", artifacts["nested_weight_selection"])
    write_markdown_summary(args.output_dir / f"{args.output_stem}_comparison.md", artifacts["group_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
