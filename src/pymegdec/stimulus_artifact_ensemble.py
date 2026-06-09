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
    "confidence_weighted_mean_score",
    "entropy_weighted_mean_score",
    "agreement_weighted_mean_score",
    "log_score_mean",
    "score_rank_fusion",
    "reciprocal_rank_fusion",
    "mean_rank",
    "borda",
    "score_tiebreak_first_source",
    "balanced_assignment",
    "balanced_assignment_shrink25",
    "balanced_assignment_shrink50",
    "balanced_assignment_shrink75",
    "balanced_assignment_low_margin10",
    "balanced_assignment_low_margin20",
    "balanced_assignment_low_margin30",
    "balanced_assignment_low_margin50",
    "uniform_prior_shift",
    "uniform_prior_shift_shrink25",
    "uniform_prior_shift_shrink50",
    "uniform_prior_shift_shrink75",
)
ARTIFACT_NESTED_SELECTION_METRIC_CHOICES = (
    "balanced_accuracy",
    "balanced_accuracy_lcb",
    "balanced_accuracy_delta",
    "balanced_accuracy_delta_lcb",
    "balanced_top2_top3_rank",
    "balanced_top2_top3_rank_lcb",
    "balanced_top2_top3_rank_delta",
    "balanced_top2_top3_rank_delta_lcb",
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


def _normalize_fusion_values(values: Sequence[float]) -> list[float]:
    """Return non-negative comparable values for score/rank fusion."""

    finite_values = [float(value) for value in values]
    if not finite_values:
        return []
    if all(math.isfinite(value) and value >= 0.0 for value in finite_values):
        total = sum(finite_values)
        if total > 0.0 and math.isfinite(total):
            return [value / total for value in finite_values]
    return _normalize_score_values(finite_values, "z_softmax")


def _score_margin_confidence(values: Sequence[float]) -> float:
    """Return a label-free confidence proxy for one source's class scores."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    if len(finite) == 1:
        return abs(finite[0])
    ordered = sorted(finite, reverse=True)
    margin = ordered[0] - ordered[1]
    if not math.isfinite(margin):
        return 0.0
    return max(0.0, float(margin))


def _score_entropy_confidence(values: Sequence[float]) -> float:
    """Return a label-free confidence proxy based on normalized score entropy.

    This favors sources with concentrated class-score mass while downweighting
    nearly uniform, high-entropy score vectors.  The value is in ``[0, 1]`` for
    non-negative scores and is safe to use as a dynamic source weight.
    """

    finite = [max(0.0, float(value)) for value in values if math.isfinite(float(value))]
    total = sum(finite)
    if total <= 0.0:
        return 0.0
    probabilities = [value / total for value in finite]
    if len(probabilities) <= 1:
        return 1.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    normalized_entropy = entropy / math.log(len(probabilities))
    return min(1.0, max(0.0, 1.0 - normalized_entropy))


def _score_agreement_confidences(
    values_by_source: Sequence[Sequence[float]],
    base_weights: Sequence[float],
) -> list[float]:
    """Return label-free per-source consensus agreement confidences."""

    if not values_by_source:
        return []
    if len(values_by_source) == 1:
        return [1.0]

    probability_rows = [_normalize_fusion_values(values) for values in values_by_source]
    confidences: list[float] = []
    for source_index, values in enumerate(probability_rows):
        other_weight_total = sum(
            weight
            for index, weight in enumerate(base_weights)
            if index != source_index and math.isfinite(weight) and weight > 0.0
        )
        if other_weight_total <= 0.0:
            confidences.append(1.0)
            continue

        consensus = [
            sum(
                float(base_weights[index]) * probability_rows[index][label_index]
                for index in range(len(probability_rows))
                if index != source_index and math.isfinite(float(base_weights[index])) and float(base_weights[index]) > 0.0
            )
            / other_weight_total
            for label_index in range(len(values))
        ]
        value_norm = math.sqrt(sum(value * value for value in values))
        consensus_norm = math.sqrt(sum(value * value for value in consensus))
        if value_norm <= 1e-12 or consensus_norm <= 1e-12:
            confidences.append(0.0)
            continue
        agreement = sum(value * consensus_value for value, consensus_value in zip(values, consensus, strict=True))
        confidences.append(min(1.0, max(0.0, agreement / (value_norm * consensus_norm))))

    return confidences if any(confidence > 0.0 for confidence in confidences) else [1.0 for _ in probability_rows]


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
        "confidence_weighted": "confidence_weighted_mean_score",
        "confidence_weighted_score": "confidence_weighted_mean_score",
        "confidence_weighted_score_mean": "confidence_weighted_mean_score",
        "margin_weighted_score_mean": "confidence_weighted_mean_score",
        "rank": "mean_rank",
        "rank_mean": "mean_rank",
        "entropy_weighted": "entropy_weighted_mean_score",
        "entropy_weighted_mean": "entropy_weighted_mean_score",
        "entropy_weighted_score": "entropy_weighted_mean_score",
        "entropy_weighted_score_mean": "entropy_weighted_mean_score",
        "agreement_weighted": "agreement_weighted_mean_score",
        "agreement_weighted_score": "agreement_weighted_mean_score",
        "agreement_weighted_score_mean": "agreement_weighted_mean_score",
        "consensus_weighted_score_mean": "agreement_weighted_mean_score",
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
        "score_rank": "score_rank_fusion",
        "rank_score": "score_rank_fusion",
        "score_rank_mean": "score_rank_fusion",
        "rank_score_mean": "score_rank_fusion",
        "score_borda_fusion": "score_rank_fusion",
        "reciprocal_rank": "reciprocal_rank_fusion",
        "reciprocal_rank_mean": "reciprocal_rank_fusion",
        "rank_reciprocal": "reciprocal_rank_fusion",
        "rank_rrf": "reciprocal_rank_fusion",
        "rrf": "reciprocal_rank_fusion",
        "quota": "balanced_assignment",
        "balanced": "balanced_assignment",
        "uniform_balanced_assignment": "balanced_assignment",
        "balanced_assignment_shrinkage": "balanced_assignment_shrink50",
        "balanced_assignment_shrinkage_25": "balanced_assignment_shrink25",
        "balanced_assignment_shrinkage_50": "balanced_assignment_shrink50",
        "balanced_assignment_shrinkage_75": "balanced_assignment_shrink75",
        "balanced_assignment_25": "balanced_assignment_shrink25",
        "balanced_assignment_50": "balanced_assignment_shrink50",
        "balanced_assignment_75": "balanced_assignment_shrink75",
        "quota_shrink25": "balanced_assignment_shrink25",
        "quota_shrink50": "balanced_assignment_shrink50",
        "quota_shrink75": "balanced_assignment_shrink75",
        "balanced_assignment_lm10": "balanced_assignment_low_margin10",
        "balanced_assignment_lm20": "balanced_assignment_low_margin20",
        "balanced_assignment_lm30": "balanced_assignment_low_margin30",
        "balanced_assignment_lm50": "balanced_assignment_low_margin50",
        "balanced_low_margin": "balanced_assignment_low_margin20",
        "balanced_low_margin10": "balanced_assignment_low_margin10",
        "balanced_low_margin20": "balanced_assignment_low_margin20",
        "balanced_low_margin30": "balanced_assignment_low_margin30",
        "balanced_low_margin50": "balanced_assignment_low_margin50",
        "quota_low_margin": "balanced_assignment_low_margin20",
        "low_margin_balanced_assignment": "balanced_assignment_low_margin20",
        "low_margin_quota": "balanced_assignment_low_margin20",
        "prior_shift": "uniform_prior_shift",
        "uniform_prior": "uniform_prior_shift",
        "uniform_prior_correction": "uniform_prior_shift",
        "uniform_prior_logit_shift": "uniform_prior_shift",
        "label_shift": "uniform_prior_shift",
        "uniform_prior_shift_shrinkage": "uniform_prior_shift_shrink50",
        "uniform_prior_shift_25": "uniform_prior_shift_shrink25",
        "uniform_prior_shift_50": "uniform_prior_shift_shrink50",
        "uniform_prior_shift_75": "uniform_prior_shift_shrink75",
        "prior_shift_shrink50": "uniform_prior_shift_shrink50",
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
        "balanced_delta": "balanced_accuracy_delta",
        "balanced_delta_lcb": "balanced_accuracy_delta_lcb",
        "paired_balanced": "balanced_accuracy_delta",
        "paired_balanced_lcb": "balanced_accuracy_delta_lcb",
        "paired_balanced_delta": "balanced_accuracy_delta",
        "paired_balanced_delta_lcb": "balanced_accuracy_delta_lcb",
        "balanced_rank": "balanced_top2_top3_rank",
        "balanced_rank_lcb": "balanced_top2_top3_rank_lcb",
        "rank_lcb": "balanced_top2_top3_rank_lcb",
        "balanced_rank_delta": "balanced_top2_top3_rank_delta",
        "balanced_rank_delta_lcb": "balanced_top2_top3_rank_delta_lcb",
        "rank_delta": "balanced_top2_top3_rank_delta",
        "rank_delta_lcb": "balanced_top2_top3_rank_delta_lcb",
        "paired_rank": "balanced_top2_top3_rank_delta",
        "paired_rank_lcb": "balanced_top2_top3_rank_delta_lcb",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ARTIFACT_NESTED_SELECTION_METRIC_CHOICES:
        raise ValueError(
            "Artifact nested selection metric must be one of "
            f"{', '.join(ARTIFACT_NESTED_SELECTION_METRIC_CHOICES)}."
        )
    return normalized


def _is_balanced_assignment_mode(aggregation_mode: str) -> bool:
    """Return whether an aggregation mode applies participant-level assignment."""

    return _normalize_artifact_aggregation_mode(aggregation_mode).startswith("balanced_assignment")


def _is_uniform_prior_shift_mode(aggregation_mode: str) -> bool:
    """Return whether an aggregation mode applies participant-level prior correction."""

    return _normalize_artifact_aggregation_mode(aggregation_mode).startswith("uniform_prior_shift")


def _balanced_assignment_uniform_alpha(aggregation_mode: str) -> float:
    """Return the assignment quota shrinkage toward a uniform class prior."""

    normalized = _normalize_artifact_aggregation_mode(aggregation_mode)
    if normalized == "balanced_assignment_shrink25":
        return 0.25
    if normalized == "balanced_assignment_shrink50":
        return 0.50
    if normalized == "balanced_assignment_shrink75":
        return 0.75
    return 1.0


def _balanced_assignment_low_margin_threshold(aggregation_mode: str) -> float | None:
    """Return the fixed-prediction margin threshold for low-margin assignment modes."""

    normalized = _normalize_artifact_aggregation_mode(aggregation_mode)
    if normalized == "balanced_assignment_low_margin10":
        return 0.10
    if normalized == "balanced_assignment_low_margin20":
        return 0.20
    if normalized == "balanced_assignment_low_margin30":
        return 0.30
    if normalized == "balanced_assignment_low_margin50":
        return 0.50
    return None


def _uniform_prior_shift_alpha(aggregation_mode: str) -> float:
    """Return the correction strength for participant-level uniform-prior score shifting."""

    normalized = _normalize_artifact_aggregation_mode(aggregation_mode)
    if normalized == "uniform_prior_shift_shrink25":
        return 0.25
    if normalized == "uniform_prior_shift_shrink50":
        return 0.50
    if normalized == "uniform_prior_shift_shrink75":
        return 0.75
    return 1.0


def _common_value_columns(rows: Sequence[dict[str, str]], pattern: re.Pattern[str], *, offset: int = 0) -> dict[int, str]:
    """Return value columns common to all rows for one regex pattern."""

    common: dict[int, str] | None = None
    for row in rows:
        columns: dict[int, str] = {}
        for column in row:
            if str(row.get(column, "")).strip() == "":
                continue
            match = pattern.match(column)
            if match:
                columns[int(match.group(1)) + int(offset)] = column
        common = columns if common is None else {label: column for label, column in common.items() if label in columns}
    return common or {}


def _class_value_columns(
    rows: Sequence[dict[str, str]],
    patterns: Sequence[tuple[re.Pattern[str], int]],
    *,
    class_labels: Sequence[int] | None = None,
) -> dict[int, str]:
    """Return per-class score/rank columns with label-basis inference.

    Prediction artifacts come from both the legacy nested-matrix path and newer
    latent-AE experiments.  The legacy path often emits display/stimulus columns
    such as ``score_1`` for raw class label ``0``.  Latent-AE outputs can be
    genuinely 1-based, where ``score_1`` means class label ``1``.  The old fixed
    ``score_N -> class N-1`` rule therefore silently misaligned scores when
    ensembling 1-based latent artifacts with logistic artifacts.

    Prefer explicit ``score_class_<raw_label>`` / ``rank_class_<raw_label>``
    columns whenever available.  For display-only ``score_<N>`` / ``rank_<N>``
    columns, infer whether the labels are raw or one-shifted from the ensemble's
    observed class labels.  This keeps the old zero-based stimulus behavior while
    making 1-based latent outputs first-class artifact-ensemble inputs.
    """

    if class_labels is not None and len(patterns) >= 2:
        labels = tuple(int(label) for label in class_labels)
        label_set = set(labels)

        # The first pattern is the raw-label form: score_class_*/rank_class_*.
        explicit_columns = _common_value_columns(rows, patterns[0][0], offset=0)
        if explicit_columns:
            return {label: explicit_columns[label] for label in labels if label in explicit_columns}

        # The second pattern is the display/stimulus form: score_*/rank_*.
        display_pattern = patterns[1][0]
        direct_columns = _common_value_columns(rows, display_pattern, offset=0)
        shifted_columns = _common_value_columns(rows, display_pattern, offset=-1)
        if direct_columns or shifted_columns:
            direct_hits = len(label_set.intersection(direct_columns))
            shifted_hits = len(label_set.intersection(shifted_columns))
            if label_set and label_set.issubset(direct_columns):
                chosen = direct_columns
            elif label_set and label_set.issubset(shifted_columns):
                chosen = shifted_columns
            elif direct_hits > shifted_hits:
                chosen = direct_columns
            else:
                # Preserve the historical zero-based stimulus-column fallback on
                # ties or when the observed label set is incomplete.
                chosen = shifted_columns
            return {label: chosen[label] for label in labels if label in chosen}

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
    confidence_weighted: bool = False,
    entropy_weighted: bool = False,
    agreement_weighted: bool = False,
) -> tuple[dict[int, float], str] | None:
    score_columns = _class_value_columns(source_rows, CLASS_SCORE_PATTERNS, class_labels=class_labels)
    if not score_columns:
        return None
    normalized = _normalize_artifact_score_normalization(score_normalization)
    weights = _normalized_source_weights(source_weights, len(source_rows))
    scored_labels = [label for label in class_labels if label in score_columns]
    if sum((confidence_weighted, entropy_weighted, agreement_weighted)) > 1:
        raise ValueError("Artifact score aggregation can use only one dynamic source-weighting mode at a time.")
    if not scored_labels:
        return None
    normalized_rows: list[tuple[list[float], float]] = []
    for row, base_weight in zip(source_rows, weights, strict=True):
        values = [float(row[score_columns[label]]) for label in scored_labels]
        normalized_values = _normalize_score_values(values, normalized)
        confidence = _score_margin_confidence(normalized_values) if confidence_weighted else 1.0
        if entropy_weighted:
            confidence = _score_entropy_confidence(normalized_values)
        normalized_rows.append((normalized_values, base_weight * confidence))

    if agreement_weighted:
        agreement_confidences = _score_agreement_confidences([values for values, _weight in normalized_rows], weights)
        normalized_rows = [
            (values, weight * confidence)
            for (values, weight), confidence in zip(normalized_rows, agreement_confidences, strict=True)
        ]

    if confidence_weighted or entropy_weighted or agreement_weighted:
        dynamic_total = sum(weight for _values, weight in normalized_rows if math.isfinite(weight) and weight > 0.0)
        if dynamic_total > 0.0:
            dynamic_weights = [max(0.0, weight) / dynamic_total for _values, weight in normalized_rows]
        else:
            dynamic_weights = list(weights)
    else:
        dynamic_weights = list(weights)

    scores: dict[int, float] = {}
    for (normalized_values, _confidence_weight), weight in zip(normalized_rows, dynamic_weights, strict=True):
        for label, value in zip(scored_labels, normalized_values, strict=True):
            contribution = _safe_log_score(value) if use_log_scores else value
            scores[label] = scores.get(label, 0.0) + weight * contribution
    if agreement_weighted and normalized == "raw" and source_weights is None:
        source = "class_score_agreement_weighted_mean"
    elif agreement_weighted and normalized == "raw":
        source = "class_score_prior_agreement_weighted_mean"
    elif agreement_weighted and source_weights is not None:
        source = f"class_score_{normalized}_prior_agreement_weighted_mean"
    elif agreement_weighted:
        source = f"class_score_{normalized}_agreement_weighted_mean"
    elif entropy_weighted and normalized == "raw" and source_weights is None:
        source = "class_score_entropy_weighted_mean"
    elif entropy_weighted and normalized == "raw":
        source = "class_score_prior_entropy_weighted_mean"
    elif entropy_weighted and source_weights is not None:
        source = f"class_score_{normalized}_prior_entropy_weighted_mean"
    elif entropy_weighted:
        source = f"class_score_{normalized}_entropy_weighted_mean"
    elif confidence_weighted and normalized == "raw" and source_weights is None:
        source = "class_score_confidence_weighted_mean"
    elif confidence_weighted and normalized == "raw":
        source = "class_score_prior_confidence_weighted_mean"
    elif confidence_weighted and source_weights is not None:
        source = f"class_score_{normalized}_prior_confidence_weighted_mean"
    elif confidence_weighted:
        source = f"class_score_{normalized}_confidence_weighted_mean"
    elif use_log_scores and normalized == "raw" and source_weights is None:
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


def _class_rank_values(row: dict[str, str], class_labels: Sequence[int]) -> tuple[dict[int, float], str]:
    """Return per-class ranks from explicit rank columns or score-derived ranks."""

    rank_columns = _class_value_columns([row], CLASS_RANK_PATTERNS, class_labels=class_labels)
    ranks: dict[int, float] = {}
    for label in class_labels:
        column = rank_columns.get(label)
        if column is None:
            continue
        value = _to_float(row[column])
        if math.isfinite(value):
            ranks[label] = value
    if ranks:
        return ranks, "rank"

    score_columns = _class_value_columns([row], CLASS_SCORE_PATTERNS, class_labels=class_labels)
    scored: list[tuple[int, float]] = []
    for label in class_labels:
        column = score_columns.get(label)
        if column is None:
            continue
        value = _to_float(row[column])
        if math.isfinite(value):
            scored.append((label, value))
    if not scored:
        return {}, ""

    ordered = sorted(scored, key=lambda item: (-item[1], item[0]))
    return {
        label: float(rank)
        for rank, (label, _score) in enumerate(ordered, start=1)
    }, "score"


def _aggregate_class_ranks(
    source_rows: Sequence[dict[str, str]],
    *,
    class_labels: Sequence[int],
    source_weights: Sequence[float] | None = None,
) -> tuple[dict[int, float], str] | None:
    weights = _normalized_source_weights(source_weights, len(source_rows))
    rank_scores: dict[int, float] = {}
    rank_sources: set[str] = set()
    for row, weight in zip(source_rows, weights, strict=True):
        rank_values, rank_source = _class_rank_values(row, class_labels)
        if not rank_values:
            continue
        rank_sources.add(rank_source)
        for label, rank in rank_values.items():
            rank_scores[label] = rank_scores.get(label, 0.0) - weight * rank
    if not rank_scores:
        return None
    if rank_sources == {"score"}:
        source = "class_score_derived_rank_mean"
    elif rank_sources == {"rank"}:
        source = "class_rank_mean"
    else:
        source = "class_mixed_rank_mean"
    return rank_scores, source


def _aggregate_reciprocal_rank_scores(
    source_rows: Sequence[dict[str, str]],
    *,
    class_labels: Sequence[int],
    source_weights: Sequence[float] | None = None,
) -> tuple[dict[int, float], str] | None:
    """Aggregate ranks as weighted reciprocal-rank scores."""

    weights = _normalized_source_weights(source_weights, len(source_rows))
    reciprocal_scores: dict[int, float] = {}
    rank_sources: set[str] = set()
    for row, weight in zip(source_rows, weights, strict=True):
        rank_values, rank_source = _class_rank_values(row, class_labels)
        if not rank_values:
            continue
        rank_sources.add(rank_source)
        for label, rank in rank_values.items():
            reciprocal_scores[label] = reciprocal_scores.get(label, 0.0) + weight / max(float(rank), 1e-12)
    if not reciprocal_scores:
        return None
    if rank_sources == {"score"}:
        source = "class_score_derived_reciprocal_rank_fusion"
    elif rank_sources == {"rank"}:
        source = "class_reciprocal_rank_fusion"
    else:
        source = "class_mixed_reciprocal_rank_fusion"
    return reciprocal_scores, source


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
    score_modes = {
        "auto",
        "mean_score",
        "confidence_weighted_mean_score",
        "entropy_weighted_mean_score",
        "agreement_weighted_mean_score",
        "log_score_mean",
        "score_tiebreak_first_source",
    }
    score_modes = {
        *score_modes,
        *(
            mode
            for mode in ARTIFACT_AGGREGATION_MODE_CHOICES
            if _is_balanced_assignment_mode(mode) or _is_uniform_prior_shift_mode(mode)
        ),
    }
    aggregated = None
    if mode in score_modes:
        aggregated = _aggregate_class_scores(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
            score_normalization=score_normalization,
            use_log_scores=mode == "log_score_mean",
            confidence_weighted=mode == "confidence_weighted_mean_score",
            entropy_weighted=mode == "entropy_weighted_mean_score",
            agreement_weighted=mode == "agreement_weighted_mean_score",
        )
    if mode == "score_rank_fusion":
        score_aggregated = _aggregate_class_scores(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
            score_normalization=score_normalization,
        )
        rank_aggregated = _aggregate_class_ranks(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
        )
        if score_aggregated is None or rank_aggregated is None:
            return None
        scores, score_source = score_aggregated
        rank_scores, rank_source = rank_aggregated
        fused_labels = [
            label for label in class_labels if label in scores and label in rank_scores
        ]
        if not fused_labels:
            return None
        score_values = _normalize_fusion_values([scores[label] for label in fused_labels])
        rank_values = _normalize_score_values(
            [rank_scores[label] for label in fused_labels],
            "z_softmax",
        )
        fused_scores = {
            label: 0.5 * score_value + 0.5 * rank_value
            for label, score_value, rank_value in zip(
                fused_labels,
                score_values,
                rank_values,
                strict=True,
            )
        }
        tie_order = {label: index for index, label in enumerate(tie_break_labels)}
        return (
            sorted(
                class_labels,
                key=lambda label: (
                    -fused_scores.get(label, float("-inf")),
                    tie_order.get(label, len(tie_order)),
                    label,
                ),
            ),
            f"{score_source}_{rank_source}_fusion",
        )
    if mode == "reciprocal_rank_fusion":
        reciprocal_aggregated = _aggregate_reciprocal_rank_scores(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
        )
        if reciprocal_aggregated is None:
            return None
        reciprocal_scores, reciprocal_source = reciprocal_aggregated
        tie_order = {label: index for index, label in enumerate(tie_break_labels)}
        return (
            sorted(
                class_labels,
                key=lambda label: (
                    -reciprocal_scores.get(label, float("-inf")),
                    tie_order.get(label, len(tie_order)),
                    label,
                ),
            ),
            reciprocal_source,
        )
    if aggregated is not None:
        scores, source = aggregated
        if mode == "score_tiebreak_first_source":
            source = f"{source}_tiebreak_first_source"
        if _is_balanced_assignment_mode(mode):
            source = f"{source}_{mode}_candidate"
        if _is_uniform_prior_shift_mode(mode):
            source = f"{source}_{mode}_candidate"
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

    if mode in {
        "mean_score",
        "confidence_weighted_mean_score",
        "entropy_weighted_mean_score",
        "agreement_weighted_mean_score",
        "log_score_mean",
        "score_tiebreak_first_source",
    }:
        return None

    if mode in {"auto", "mean_rank", "borda"}:
        rank_aggregated = _aggregate_class_ranks(
            source_rows,
            class_labels=class_labels,
            source_weights=source_weights,
        )
    else:
        rank_aggregated = None
    if rank_aggregated is not None:
        rank_scores, rank_source = rank_aggregated
        if rank_scores:
            if mode == "borda":
                if rank_source == "class_score_derived_rank_mean":
                    rank_source = "class_score_derived_rank_borda"
                elif rank_source == "class_mixed_rank_mean":
                    rank_source = "class_mixed_rank_borda"
                else:
                    rank_source = "class_rank_borda"
            return (
                sorted(
                    class_labels,
                    key=lambda label: (-rank_scores.get(label, float("-inf")), label),
                ),
                rank_source,
            )

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


def _scores_are_probability_like(scores: dict[int, float], class_labels: Sequence[int]) -> bool:
    """Return whether a complete score vector is safe to expose as probabilities."""

    values: list[float] = []
    for label in class_labels:
        if label not in scores:
            return False
        value = float(scores[label])
        if not math.isfinite(value) or value < -1e-9:
            return False
        values.append(max(0.0, value))
    return math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def _add_score_alias_columns(
    row: dict[str, object],
    *,
    scores: dict[int, float],
    class_labels: Sequence[int],
    display_labels: dict[int, int],
) -> None:
    """Expose artifact ensemble scores in the standard prediction schema.

    ``artifact_score_class_*`` columns are retained for backward compatibility,
    while ``score_class_*`` / ``score_*`` make artifact-ensemble outputs usable as
    first-class inputs for later score-level artifact ensembling.  Probability
    aliases are emitted only when the complete class vector is non-negative and
    sums to one, which avoids labelling log-score or z-normalized values as
    probabilities.
    """

    probability_like = _scores_are_probability_like(scores, class_labels)
    for label in class_labels:
        if label not in scores:
            continue
        value = float(scores[label])
        display_label = display_labels.get(label, label)
        row[f"artifact_score_class_{label}"] = value
        row[f"score_class_{label}"] = value
        row[f"score_{display_label}"] = value
        if probability_like:
            row[f"prob_class_{label}"] = value
            row[f"prob_{display_label}"] = value


def _add_rank_alias_columns(
    row: dict[str, object],
    *,
    ranked_labels: Sequence[int],
    class_labels: Sequence[int],
    display_labels: dict[int, int],
) -> None:
    """Expose per-class ranks in the standard prediction schema."""

    rank_by_label = {int(label): rank for rank, label in enumerate(ranked_labels, start=1)}
    for label in class_labels:
        rank = rank_by_label.get(int(label))
        if rank is None:
            continue
        display_label = display_labels.get(label, label)
        row[f"rank_class_{label}"] = rank
        row[f"rank_{display_label}"] = rank


def _update_rank_metrics_from_labels(
    row: dict[str, object],
    *,
    ranked_labels: Sequence[int],
    true_label: int,
    class_labels: Sequence[int],
    display_labels: dict[int, int],
) -> None:
    """Synchronize rank/top-k fields after a post-hoc prediction update."""

    true_rank = float(list(ranked_labels).index(true_label) + 1)
    row["true_label_rank"] = true_rank
    row["top2_correct"] = true_rank <= 2
    row["top3_correct"] = true_rank <= 3
    row["vote_ranked_labels"] = ";".join(str(value) for value in ranked_labels)
    _add_rank_alias_columns(
        row,
        ranked_labels=ranked_labels,
        class_labels=class_labels,
        display_labels=display_labels,
    )


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
    if rank_source != "source_true_label_rank":
        _add_rank_alias_columns(
            row,
            ranked_labels=ranked_labels,
            class_labels=class_labels,
            display_labels=display_labels,
        )

    for column, value in zip(key_columns, key, strict=True):
        if column not in row:
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
        confidence_weighted=aggregation_mode == "confidence_weighted_mean_score",
        entropy_weighted=aggregation_mode == "entropy_weighted_mean_score",
        agreement_weighted=aggregation_mode == "agreement_weighted_mean_score",
    )
    if aggregated is not None:
        aggregate_scores, _score_source = aggregated
        _add_score_alias_columns(
            row,
            scores=aggregate_scores,
            class_labels=class_labels,
            display_labels=display_labels,
        )
    return row


def _uniform_class_quotas(n_items: int, n_classes: int) -> list[int]:
    if n_classes <= 0:
        return []
    quotas = [int(n_items) // int(n_classes) for _ in range(int(n_classes))]
    for index in range(int(n_items) - sum(quotas)):
        quotas[index] += 1
    return quotas


def _rounded_quota_counts(expected_counts: Sequence[float], *, n_items: int) -> list[int]:
    """Round expected class counts while preserving the total row count."""

    expected = [max(0.0, float(value)) for value in expected_counts]
    if not expected:
        return []
    floors = [int(math.floor(value)) for value in expected]
    remaining = int(n_items) - sum(floors)
    if remaining < 0:
        total = sum(expected)
        if total <= 0.0:
            return _uniform_class_quotas(n_items, len(expected))
        return _rounded_quota_counts(
            [value * float(n_items) / total for value in expected],
            n_items=n_items,
        )

    remainders = [value - math.floor(value) for value in expected]
    order = sorted(range(len(expected)), key=lambda index: (-remainders[index], index))
    quotas = list(floors)
    for index in order[:remaining]:
        quotas[index] += 1
    return quotas


def _balanced_assignment_class_quotas(
    rows: Sequence[dict[str, object]],
    indices: Sequence[int],
    class_labels: Sequence[int],
    *,
    uniform_alpha: float,
) -> list[int]:
    """Return class quotas for exact or shrinkage balanced assignment."""

    label_list = [int(label) for label in class_labels]
    uniform_alpha = min(max(float(uniform_alpha), 0.0), 1.0)
    uniform = _uniform_class_quotas(len(indices), len(label_list))
    if uniform_alpha >= 1.0 - 1e-12:
        return uniform

    label_to_index = {label: index for index, label in enumerate(label_list)}
    predicted_counts = [0 for _ in label_list]
    for row_index in indices:
        predicted = _to_int(rows[row_index]["predicted_label"], field="predicted_label")
        if predicted in label_to_index:
            predicted_counts[label_to_index[predicted]] += 1
    expected = [
        (1.0 - uniform_alpha) * predicted + uniform_alpha * quota
        for predicted, quota in zip(predicted_counts, uniform, strict=True)
    ]
    return _rounded_quota_counts(expected, n_items=len(indices))


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


def _low_margin_balanced_assignment_indices(
    score_matrix: Sequence[Sequence[float]],
    quotas: Sequence[int],
    *,
    margin_threshold: float,
) -> tuple[list[int], float, int]:
    """Assign only ambiguous rows while preserving high-margin argmax calls."""

    scores = [list(map(float, row)) for row in score_matrix]
    quotas = [int(quota) for quota in quotas]
    if not scores:
        return [], 0.0, 0
    if len(quotas) != len(scores[0]):
        raise ValueError("Low-margin balanced-assignment quotas must match the score width.")
    if sum(quotas) != len(scores):
        raise ValueError("Low-margin balanced-assignment quotas must sum to the number of rows.")

    argmax_indices: list[int] = []
    margins: list[float] = []
    for row in scores:
        order = sorted(range(len(row)), key=lambda class_index: (-row[class_index], class_index))
        argmax_indices.append(int(order[0]))
        if len(order) == 1:
            margins.append(math.inf)
        else:
            margins.append(float(row[order[0]]) - float(row[order[1]]))

    threshold = max(0.0, float(margin_threshold))
    fixed_mask = [margin >= threshold for margin in margins]
    for class_index, quota in enumerate(quotas):
        fixed_rows = [
            row_index
            for row_index, fixed in enumerate(fixed_mask)
            if fixed and argmax_indices[row_index] == class_index
        ]
        overflow = len(fixed_rows) - int(quota)
        if overflow > 0:
            for row_index in sorted(fixed_rows, key=lambda index: (margins[index], index))[:overflow]:
                fixed_mask[row_index] = False

    remaining_quotas = list(quotas)
    for fixed, class_index in zip(fixed_mask, argmax_indices, strict=True):
        if fixed:
            remaining_quotas[int(class_index)] -= 1
    if any(quota < 0 for quota in remaining_quotas):
        raise ValueError("Low-margin balanced assignment produced negative remaining quotas.")

    predicted = list(argmax_indices)
    remaining_rows = [row_index for row_index, fixed in enumerate(fixed_mask) if not fixed]
    if remaining_rows:
        remaining_scores = [scores[row_index] for row_index in remaining_rows]
        assigned_remaining, _remaining_objective_delta = _balanced_assignment_indices(remaining_scores, remaining_quotas)
        for row_index, assigned_index in zip(remaining_rows, assigned_remaining, strict=True):
            predicted[row_index] = int(assigned_index)

    argmax_score = sum(max(row) for row in scores)
    assignment_score = sum(scores[row_index][class_index] for row_index, class_index in enumerate(predicted))
    return predicted, float(assignment_score - argmax_score), int(sum(fixed_mask))


def _rank_labels_with_forced_first_label(
    row: dict[str, object],
    class_labels: Sequence[int],
    first_label: int,
) -> list[int]:
    """Return a full ranking with a post-hoc assigned class in first position."""

    score_by_label: dict[int, float] = {}
    for label in class_labels:
        value = str(row.get(f"artifact_score_class_{label}", "")).strip()
        if value == "":
            continue
        score_by_label[int(label)] = _to_float(value)
    remaining = [int(label) for label in class_labels if int(label) != int(first_label)]
    remaining = sorted(
        remaining,
        key=lambda label: (-score_by_label.get(label, float("-inf")), label),
    )
    return [int(first_label), *remaining]


def _apply_balanced_assignment_rows(
    prediction_rows: Sequence[dict[str, object]],
    class_labels: Sequence[int],
    *,
    uniform_alpha: float = 1.0,
    margin_threshold: float | None = None,
) -> list[dict[str, object]]:
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
        quotas = _balanced_assignment_class_quotas(
            rows,
            indices,
            label_list,
            uniform_alpha=uniform_alpha,
        )
        if margin_threshold is None:
            assigned_indices, objective_delta = _balanced_assignment_indices(score_matrix, quotas)
            fixed_predictions = 0
        else:
            assigned_indices, objective_delta, fixed_predictions = _low_margin_balanced_assignment_indices(
                score_matrix,
                quotas,
                margin_threshold=float(margin_threshold),
            )
        quota_text = ";".join(f"{label}:{quota}" for label, quota in zip(label_list, quotas, strict=True))
        mode_suffix = "" if margin_threshold is None else f"_low_margin{int(round(100.0 * float(margin_threshold))):02d}"
        for row_index, assigned_index in zip(indices, assigned_indices, strict=True):
            row = rows[row_index]
            predicted_label = label_list[int(assigned_index)]
            true_label = _to_int(row["true_label"], field="true_label")
            row["predicted_label"] = predicted_label
            row["predicted_stimulus"] = display_labels.get(predicted_label, predicted_label)
            row["correct"] = predicted_label == true_label
            row["artifact_ensemble_mode"] = (
                "class_score_balanced_assignment"
                if uniform_alpha >= 1.0 - 1e-12
                else f"class_score_balanced_assignment_shrink{int(round(100.0 * uniform_alpha)):02d}"
            ) + mode_suffix
            row["artifact_ensemble_balanced_assignment_uniform_alpha"] = f"{uniform_alpha:.6g}"
            row["artifact_ensemble_balanced_assignment_margin_threshold"] = "" if margin_threshold is None else f"{float(margin_threshold):.6g}"
            row["artifact_ensemble_balanced_assignment_fixed_predictions"] = fixed_predictions
            row["artifact_ensemble_balanced_assignment_quota_counts"] = quota_text
            row["artifact_ensemble_balanced_assignment_objective_delta"] = objective_delta
            row["artifact_ensemble_rank_source"] = row["artifact_ensemble_mode"]
            ranked_labels = _rank_labels_with_forced_first_label(
                row,
                label_list,
                predicted_label,
            )
            _update_rank_metrics_from_labels(
                row,
                ranked_labels=ranked_labels,
                true_label=true_label,
                class_labels=label_list,
                display_labels=display_labels,
            )
    return rows


def _apply_uniform_prior_shift_rows(
    prediction_rows: Sequence[dict[str, object]],
    class_labels: Sequence[int],
    *,
    alpha: float,
) -> list[dict[str, object]]:
    """Apply label-free participant-level correction toward a uniform class prior."""

    rows = [dict(row) for row in prediction_rows]
    label_list = [int(label) for label in class_labels]
    if not rows or not label_list:
        return rows
    if alpha <= 0.0 or not math.isfinite(float(alpha)):
        raise ValueError("Uniform-prior shift alpha must be a finite positive value.")

    display_labels = _display_label_map(label_list)
    by_participant: dict[str, list[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        by_participant[str(row.get("test_participant", ""))].append(row_index)

    score_columns = [f"artifact_score_class_{label}" for label in label_list]
    for participant, indices in by_participant.items():
        missing = [
            column
            for column in score_columns
            if any(str(rows[index].get(column, "")).strip() == "" for index in indices)
        ]
        if missing:
            raise ValueError(
                "uniform_prior_shift requires class score/probability columns for every class; "
                f"participant={participant!r}, missing examples={missing[:5]}."
            )

        probabilities_by_row: list[list[float]] = []
        for row_index in indices:
            values = [_to_float(rows[row_index][column]) for column in score_columns]
            probabilities = _normalize_fusion_values(values)
            if len(probabilities) != len(label_list):
                raise ValueError("uniform_prior_shift could not normalize the complete class score vector.")
            probabilities_by_row.append(probabilities)

        mean_probabilities = [
            sum(probabilities[label_index] for probabilities in probabilities_by_row) / len(probabilities_by_row)
            for label_index in range(len(label_list))
        ]
        prior = 1.0 / len(label_list)
        multipliers = [
            (prior / max(mean_probability, 1e-12)) ** float(alpha)
            for mean_probability in mean_probabilities
        ]
        mean_text = ";".join(
            f"{label}:{mean_probability:.6g}"
            for label, mean_probability in zip(label_list, mean_probabilities, strict=True)
        )
        mode = (
            "class_score_uniform_prior_shift"
            if alpha >= 1.0 - 1e-12
            else f"class_score_uniform_prior_shift_shrink{int(round(100.0 * float(alpha))):02d}"
        )

        for row_index, probabilities in zip(indices, probabilities_by_row, strict=True):
            row = rows[row_index]
            adjusted_values = [
                probability * multiplier
                for probability, multiplier in zip(probabilities, multipliers, strict=True)
            ]
            total = sum(adjusted_values)
            if total <= 0.0 or not math.isfinite(total):
                adjusted_values = [prior for _ in label_list]
                total = 1.0
            adjusted_scores = {
                label: max(0.0, adjusted_value) / total
                for label, adjusted_value in zip(label_list, adjusted_values, strict=True)
            }
            ranked_labels = sorted(label_list, key=lambda label: (-adjusted_scores.get(label, 0.0), label))
            predicted_label = int(ranked_labels[0])
            true_label = _to_int(row["true_label"], field="true_label")
            row["predicted_label"] = predicted_label
            row["predicted_stimulus"] = display_labels.get(predicted_label, predicted_label)
            row["correct"] = predicted_label == true_label
            row["artifact_ensemble_mode"] = mode
            row["artifact_ensemble_uniform_prior_shift_alpha"] = f"{float(alpha):.6g}"
            row["artifact_ensemble_uniform_prior_shift_mean_scores"] = mean_text
            row["artifact_ensemble_rank_source"] = mode
            _add_score_alias_columns(
                row,
                scores=adjusted_scores,
                class_labels=label_list,
                display_labels=display_labels,
            )
            _update_rank_metrics_from_labels(
                row,
                ranked_labels=ranked_labels,
                true_label=true_label,
                class_labels=label_list,
                display_labels=display_labels,
            )
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


def _nested_selection_metric_base(selection_metric: str) -> str:
    """Return the non-delta metric family used for one outer-row score."""

    normalized = _normalize_artifact_nested_selection_metric(selection_metric)
    if normalized.startswith("balanced_accuracy"):
        return "balanced_accuracy"
    return "balanced_top2_top3_rank"


def _nested_selection_uses_lcb(selection_metric: str) -> bool:
    return _normalize_artifact_nested_selection_metric(selection_metric).endswith("_lcb")


def _nested_selection_is_delta(selection_metric: str) -> bool:
    return "_delta" in _normalize_artifact_nested_selection_metric(selection_metric)


def _nested_selection_row_score(row: dict[str, object], *, selection_metric: str, n_classes: int) -> float:
    base = _nested_selection_metric_base(selection_metric)
    if base == "balanced_accuracy":
        return _to_float(row["balanced_accuracy"])
    return _artifact_recipe_rank_score(row, n_classes=n_classes)


def _outer_rows_by_participant(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row.get("test_participant", "")): row for row in rows}


def _nested_selection_metric_value(
    rows: Sequence[dict[str, object]],
    *,
    selection_metric: str,
    n_classes: int,
    reference_rows: Sequence[dict[str, object]] | None = None,
) -> float:
    normalized = _normalize_artifact_nested_selection_metric(selection_metric)
    if _nested_selection_is_delta(normalized):
        if reference_rows is None:
            raise ValueError(f"Nested selection metric {normalized!r} requires reference rows.")
        reference_by_participant = _outer_rows_by_participant(reference_rows)
        paired_deltas: list[float] = []
        for row in rows:
            participant = str(row.get("test_participant", ""))
            reference = reference_by_participant.get(participant)
            if reference is None:
                continue
            paired_deltas.append(
                _nested_selection_row_score(row, selection_metric=normalized, n_classes=n_classes)
                - _nested_selection_row_score(reference, selection_metric=normalized, n_classes=n_classes)
            )
        if not paired_deltas:
            return float("-inf")
        score = _mean(paired_deltas)
        if _nested_selection_uses_lcb(normalized):
            score -= _sem(paired_deltas)
        return score

    values = [
        _nested_selection_row_score(row, selection_metric=normalized, n_classes=n_classes)
        for row in rows
    ]
    score = _mean(values)
    if _nested_selection_uses_lcb(normalized):
        score -= _sem(values)
    return score


def _counts_text(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ";".join(f"{value}:{counts[value]}" for value in sorted(counts, key=_participant_sort_key))


def _resolve_nested_weight_selector_ensembles(
    ensemble_sources: dict[str, Sequence[str]],
    raw_ensemble_spec: str | None,
) -> list[str]:
    """Return multi-source ensembles requested for nested weight selection.

    The historical behavior selected the first multi-source recipe when no
    explicit ensemble was passed.  The new ``all``/comma-list syntax lets the
    artifact workflow evaluate leakage-safe source-weight grids for every
    promising recipe in one invocation, without changing the old default API
    behavior for programmatic callers.
    """

    multi_source_ensembles = [
        name
        for name, source_names in ensemble_sources.items()
        if len(source_names) > 1
    ]
    if not multi_source_ensembles:
        raise ValueError("Nested weight selection requires at least one multi-source ensemble.")

    raw = "" if raw_ensemble_spec is None else str(raw_ensemble_spec).strip()
    if raw == "":
        return [multi_source_ensembles[0]]

    requested = [token.strip() for token in raw.split(",") if token.strip()]
    if any(token.lower() in {"all", "*"} for token in requested):
        if len(requested) != 1:
            raise ValueError(
                "Use 'all' by itself for nested source-weight selection, "
                "not mixed with explicit ensemble names."
            )
        return multi_source_ensembles

    missing = [name for name in requested if name not in ensemble_sources]
    if missing:
        raise ValueError(f"Unknown nested weight-selector ensemble(s): {', '.join(missing)}")
    single_source = [name for name in requested if len(ensemble_sources[name]) < 2]
    if single_source:
        raise ValueError(
            "Nested weight selection requires multi-source ensembles; "
            f"got single-source {', '.join(single_source)}"
        )
    return list(dict.fromkeys(requested))


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
    nested_selection_metric: str,
) -> tuple[list[dict], list[dict], list[dict], dict[str, object]]:
    """Choose source weights per held-out subject using other subjects only."""

    nested_selection_metric = _normalize_artifact_nested_selection_metric(nested_selection_metric)
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
        if _is_balanced_assignment_mode(aggregation_mode):
            prediction_rows = _apply_balanced_assignment_rows(
                prediction_rows,
                class_labels,
                uniform_alpha=_balanced_assignment_uniform_alpha(aggregation_mode),
                margin_threshold=_balanced_assignment_low_margin_threshold(aggregation_mode),
            )
        if _is_uniform_prior_shift_mode(aggregation_mode):
            prediction_rows = _apply_uniform_prior_shift_rows(
                prediction_rows,
                class_labels,
                alpha=_uniform_prior_shift_alpha(aggregation_mode),
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
    reference_candidate = min(candidates, key=lambda candidate: (candidate.uniform_distance, candidate.candidate_index))

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
            reference_train_outer_rows = [
                row
                for other_participant, row in reference_candidate.outer_by_participant.items()
                if other_participant != participant
            ]
            selection_score = _nested_selection_metric_value(
                train_outer_rows,
                selection_metric=nested_selection_metric,
                n_classes=n_classes,
                reference_rows=reference_train_outer_rows,
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
                "reference_source_weights": reference_candidate.weight_text,
                "reference_weight_grid_step": grid_step,
                "reference_weight_candidate_index": reference_candidate.candidate_index,
                "selected_weight_candidate_index": selected_candidate.candidate_index,
                "selection_metric": _nested_selection_metric_label(nested_selection_metric),
                "selection_metric_name": nested_selection_metric,
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
            selected_row["reference_source_weights"] = reference_candidate.weight_text
            selected_row["reference_weight_grid_step"] = grid_step
            selected_row["reference_weight_candidate_index"] = reference_candidate.candidate_index
            selected_row["selected_weight_candidate_index"] = selected_candidate.candidate_index
            selected_row["selection_metric"] = _nested_selection_metric_label(nested_selection_metric)
            selected_row["selection_metric_name"] = nested_selection_metric
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
    summary["selection_metric"] = _nested_selection_metric_label(nested_selection_metric)
    summary["selection_metric_name"] = nested_selection_metric
    summary["selected_source_weight_counts"] = _counts_text(str(row["selected_source_weights"]) for row in selection_rows)
    summary["candidate_source_weight_count"] = len(candidates)
    summary["selected_weight_grid_step"] = grid_step
    summary["reference_source_weights"] = reference_candidate.weight_text
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
    reference_ensemble = ensemble_order[0]
    if reference_ensemble not in outer_by_ensemble_participant:
        raise ValueError(
            f"Nested subject selector reference ensemble {reference_ensemble!r} has no outer rows."
        )
    reference_outer_by_participant = outer_by_ensemble_participant[reference_ensemble]

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
            reference_train_outer_rows = [
                row
                for other_participant, row in reference_outer_by_participant.items()
                if other_participant != participant
            ]
            selection_score = _nested_selection_metric_value(
                train_outer_rows, selection_metric=nested_selection_metric, n_classes=n_classes, reference_rows=reference_train_outer_rows
            )
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
                "reference_artifact_ensemble": reference_ensemble,
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
            selected_row["reference_artifact_ensemble"] = reference_ensemble
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
    summary["reference_artifact_ensemble"] = reference_ensemble
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
) -> dict[str, list[dict]]:
    """Build leakage-safe artifact ensembles from completed prediction CSVs."""

    score_normalization = _normalize_artifact_score_normalization(score_normalization)
    aggregation_mode = _normalize_artifact_aggregation_mode(aggregation_mode)
    nested_selection_metric = _normalize_artifact_nested_selection_metric(nested_selection_metric)
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
        if _is_balanced_assignment_mode(aggregation_mode):
            prediction_rows = _apply_balanced_assignment_rows(
                prediction_rows,
                class_labels,
                uniform_alpha=_balanced_assignment_uniform_alpha(aggregation_mode),
                margin_threshold=_balanced_assignment_low_margin_threshold(aggregation_mode),
            )
        if _is_uniform_prior_shift_mode(aggregation_mode):
            prediction_rows = _apply_uniform_prior_shift_rows(
                prediction_rows,
                class_labels,
                alpha=_uniform_prior_shift_alpha(aggregation_mode),
            )
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
        selector_ensembles = _resolve_nested_weight_selector_ensembles(
            ensemble_sources,
            nested_weight_selector_ensemble,
        )
        nested_weight_selection_rows: list[dict] = []
        for selector_ensemble in selector_ensembles:
            selector_name = nested_weight_selector_name
            if len(selector_ensembles) > 1:
                selector_name = f"{nested_weight_selector_name}_{selector_ensemble}"
            weight_predictions, weight_outer, weight_selection, weight_summary = _nested_source_weight_selector(
                selector_name=selector_name,
                selector_ensemble=selector_ensemble,
                ensemble_sources=ensemble_sources,
                indexed_sources=indexed_sources,
                key_columns=key_columns,
                class_labels=class_labels,
                n_classes=len(class_labels),
                score_normalization=score_normalization,
                aggregation_mode=aggregation_mode,
                grid_step=nested_weight_grid_step,
                nested_selection_metric=nested_selection_metric,
            )
            artifacts["predictions"].extend(weight_predictions)
            artifacts["outer"].extend(weight_outer)
            artifacts["group_summary"].append(weight_summary)
            nested_weight_selection_rows.extend(weight_selection)
        artifacts["nested_weight_selection"] = nested_weight_selection_rows
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
        help="How to combine source predictions: score/log-score/entropy-weighted mean, rank/Borda, hard vote, balanced assignment, or automatic score-then-rank fallback.",
    )
    parser.add_argument(
        "--nested-selection-metric",
        choices=ARTIFACT_NESTED_SELECTION_METRIC_CHOICES,
        default="balanced_accuracy",
        help="Leave-subject-out metric for selecting artifact ensemble recipes and nested source-weight candidates.",
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
        help=(
            "Ensemble(s) whose source weights should be grid-selected by "
            "--nested-weight-selector-name. Use a comma-separated list or 'all'. "
            "Defaults to the first multi-source ensemble."
        ),
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
