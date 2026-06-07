"""Leakage-safe weighted score selection across artifact-ensemble outputs.

This module is a score-level counterpart to ``artifact_ensemble_meta_selector``.
Instead of selecting one artifact candidate for each held-out participant, it
uses the other participants to derive softmax weights over artifact candidates
and averages per-class scores for the held-out participant.  It is intended for
prediction artifacts that include ``score_class_*`` / ``score_*`` columns.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from pymegdec.artifact_ensemble_meta_selector import (
    MetaCandidate,
    _class_labels_from_rows,
    _expand_input_paths,
    _label_from_row,
    _metric_mean,
    _nested_selection_metric_label,
    _nested_selection_metric_value,
    _normalize_nested_selection_metric,
    _outer_rows,
    _participant_sort_key,
    _summary_row,
    _to_float,
    load_meta_candidates,
    write_csv_rows,
    write_markdown_summary,
)
from pymegdec.stimulus_artifact_ensemble import ARTIFACT_NESTED_SELECTION_METRIC_CHOICES


def _display_label_map(class_labels: Sequence[int]) -> dict[int, int]:
    labels = [int(label) for label in class_labels]
    if labels and min(labels) == 0 and max(labels) == len(labels) - 1:
        return {label: label + 1 for label in labels}
    return {label: label for label in labels}


def _score_value(row: dict[str, object], label: int) -> float:
    display_label = label + 1
    for column in (
        f"score_class_{label}",
        f"prob_class_{label}",
        f"score_{display_label}",
        f"prob_{display_label}",
    ):
        value = _to_float(row.get(column, ""))
        if math.isfinite(value):
            return value
    return math.nan


def _row_score_vector(row: dict[str, object], *, class_labels: Sequence[int]) -> dict[int, float] | None:
    scores: dict[int, float] = {}
    for label in class_labels:
        value = _score_value(row, int(label))
        if not math.isfinite(value):
            return None
        scores[int(label)] = value
    return scores


def _scores_are_probability_like(scores: dict[int, float], class_labels: Sequence[int]) -> bool:
    values: list[float] = []
    for label in class_labels:
        if label not in scores:
            return False
        value = float(scores[int(label)])
        if not math.isfinite(value) or value < -1e-9:
            return False
        values.append(max(value, 0.0))
    return math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def _ranked_labels_from_scores(scores: dict[int, float], class_labels: Sequence[int]) -> list[int]:
    return sorted(class_labels, key=lambda label: (-scores.get(int(label), float("-inf")), int(label)))


def _prediction_key(row: dict[str, object], *, row_index: int) -> str:
    parts: list[str] = []
    for column in ("test_trial_index", "trial", "test_trial_number"):
        value = str(row.get(column, "")).strip()
        if value:
            parts.append(f"{column}={value}")
    return "|".join(parts) if parts else f"row_index={row_index:06d}"


def _rows_by_prediction_key(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    keyed: dict[str, dict[str, object]] = {}
    for row_index, row in enumerate(rows):
        key = _prediction_key(row, row_index=row_index)
        if key in keyed:
            key = f"{key}|row_index={row_index:06d}"
        keyed[key] = row
    return keyed


def _clear_score_rank_columns(row: dict[str, object], *, class_labels: Sequence[int]) -> None:
    display_labels = _display_label_map(class_labels)
    for label in class_labels:
        display_label = display_labels.get(int(label), int(label))
        for prefix in ("score_class", "prob_class", "rank_class"):
            row.pop(f"{prefix}_{label}", None)
        for prefix in ("score", "prob", "rank"):
            row.pop(f"{prefix}_{display_label}", None)


def _format_candidate_weights(candidate_weights: Sequence[tuple[MetaCandidate, float]]) -> str:
    return ";".join(f"{candidate.name}:{weight:.6g}" for candidate, weight in candidate_weights)


def _softmax_candidate_weights(
    scored_candidates: Sequence[tuple[float, int, MetaCandidate, list[dict[str, object]]]],
    *,
    weight_temperature: float,
) -> list[tuple[float, int, MetaCandidate, list[dict[str, object]], float]]:
    finite = [item for item in scored_candidates if math.isfinite(item[0])]
    if not finite:
        finite = [(0.0, candidate_index, candidate, rows) for _score, candidate_index, candidate, rows in scored_candidates]
    if not finite:
        return []
    if weight_temperature <= 0.0:
        best = max(finite, key=lambda item: (item[0], -item[1]))
        return [(*item, 1.0 if item is best else 0.0) for item in finite]
    max_score = max(score for score, _candidate_index, _candidate, _rows in finite)
    raw_weights: list[float] = []
    for score, _candidate_index, _candidate, _rows in finite:
        exponent = max(min((score - max_score) / weight_temperature, 700.0), -700.0)
        raw_weights.append(math.exp(exponent))
    total = sum(raw_weights)
    if not math.isfinite(total) or total <= 0.0:
        uniform = 1.0 / len(finite)
        return [(*item, uniform) for item in finite]
    return [(*item, weight / total) for item, weight in zip(finite, raw_weights, strict=True)]


def _apply_weighted_scores_to_row(
    reference: dict[str, object],
    *,
    selector_name: str,
    class_labels: Sequence[int],
    scores: dict[int, float],
    candidate_weights: Sequence[tuple[MetaCandidate, float]],
    nested_selection_metric: str,
    selection_score: float,
) -> dict[str, object]:
    row: dict[str, object] = dict(reference)
    _clear_score_rank_columns(row, class_labels=class_labels)
    display_labels = _display_label_map(class_labels)
    ranked_labels = _ranked_labels_from_scores(scores, class_labels)
    true_label = _label_from_row(row, label_column="true_label", stimulus_column="true_stimulus", field="true_label")
    predicted_label = int(ranked_labels[0])
    true_rank = float(ranked_labels.index(true_label) + 1)

    row["source_artifact_ensemble"] = reference.get("artifact_ensemble", "")
    row["artifact_ensemble"] = selector_name
    row["artifact_ensemble_meta_selection"] = "leave_subject_out_weighted_score"
    row["selection_metric"] = _nested_selection_metric_label(nested_selection_metric)
    row["selection_metric_name"] = nested_selection_metric
    row["selection_metric_value"] = selection_score
    row["weighted_artifact_candidate_count"] = len(candidate_weights)
    row["weighted_artifact_candidate_weights"] = _format_candidate_weights(candidate_weights)
    row["true_label"] = true_label
    row["predicted_label"] = predicted_label
    row["true_stimulus"] = display_labels.get(true_label, true_label)
    row["predicted_stimulus"] = display_labels.get(predicted_label, predicted_label)
    row["correct"] = predicted_label == true_label
    row["true_label_rank"] = true_rank
    row["top2_correct"] = true_rank <= 2
    row["top3_correct"] = true_rank <= 3
    row["vote_ranked_labels"] = ";".join(str(label) for label in ranked_labels)

    probability_like = _scores_are_probability_like(scores, class_labels)
    rank_by_label = {int(label): rank for rank, label in enumerate(ranked_labels, start=1)}
    for label in class_labels:
        value = float(scores[int(label)])
        display_label = display_labels.get(int(label), int(label))
        row[f"score_class_{label}"] = value
        row[f"score_{display_label}"] = value
        if probability_like:
            row[f"prob_class_{label}"] = value
            row[f"prob_{display_label}"] = value
        row[f"rank_class_{label}"] = rank_by_label[int(label)]
        row[f"rank_{display_label}"] = rank_by_label[int(label)]
    return row


def _counts_text(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ";".join(f"{value}:{counts[value]}" for value in sorted(counts, key=_participant_sort_key))


def _candidate_class_labels(candidates: Sequence[MetaCandidate], *, n_classes: int) -> list[int]:
    labels = sorted(set().union(*(_class_labels_from_rows(candidate.rows) for candidate in candidates)))
    return labels if labels else list(range(n_classes))


def _weighted_summary_row(
    *,
    selector_name: str,
    prediction_rows: Sequence[dict[str, object]],
    outer_rows: Sequence[dict[str, object]],
    n_classes: int,
    selection_metric: str,
    selected_candidate_counts: str,
    n_candidates: int,
    weight_temperature: float,
) -> dict[str, object]:
    summary = _summary_row(
        selector_name=selector_name,
        prediction_rows=prediction_rows,
        outer_rows=outer_rows,
        n_classes=n_classes,
        selection_metric=selection_metric,
        selected_candidate_counts=selected_candidate_counts,
        n_candidates=n_candidates,
    )
    summary["artifact_ensemble_meta_selection"] = "leave_subject_out_weighted_score"
    summary["selection_weight_temperature"] = weight_temperature
    return summary


def nested_weighted_score_select_candidates(
    candidates: Sequence[MetaCandidate],
    *,
    selector_name: str = "cross_mode_weighted_score_selector",
    nested_selection_metric: str = "balanced_accuracy",
    n_classes: int,
    weight_temperature: float = 0.02,
) -> dict[str, list[dict]]:
    """Blend artifact candidates with source-subject-only softmax weights.

    For each held-out participant, candidate weights are computed from the other
    participants only.  The held-out participant's predictions are made by
    averaging the candidates' per-class scores with those weights.
    """

    nested_selection_metric = _normalize_nested_selection_metric(nested_selection_metric)
    participants = sorted(
        set().union(*(set(candidate.outer_by_participant) for candidate in candidates)),
        key=_participant_sort_key,
    )
    if not participants:
        raise ValueError("Weighted artifact meta-selection requires test_participant values.")
    class_labels = _candidate_class_labels(candidates, n_classes=n_classes)

    selected_predictions: list[dict] = []
    selection_rows: list[dict] = []
    for participant in participants:
        scored_candidates: list[tuple[float, int, MetaCandidate, list[dict[str, object]]]] = []
        for candidate_index, candidate in enumerate(candidates):
            if participant not in candidate.rows_by_participant:
                continue
            train_outer_rows = [row for other_participant, row in candidate.outer_by_participant.items() if other_participant != participant]
            if not train_outer_rows:
                continue
            selection_score = _nested_selection_metric_value(
                train_outer_rows,
                selection_metric=nested_selection_metric,
                n_classes=n_classes,
            )
            scored_candidates.append((selection_score, candidate_index, candidate, train_outer_rows))
        if not scored_candidates:
            raise ValueError(f"Cannot compute weighted artifact scores for participant {participant}; no candidate has source-subject rows.")

        weighted_candidates = _softmax_candidate_weights(scored_candidates, weight_temperature=weight_temperature)
        keyed_rows = [
            (candidate, weight, _rows_by_prediction_key(candidate.rows_by_participant[participant]), selection_score, train_outer_rows)
            for selection_score, _candidate_index, candidate, train_outer_rows, weight in weighted_candidates
            if weight > 0.0
        ]
        reference_candidate = min(scored_candidates, key=lambda item: item[1])[2]
        reference_rows = _rows_by_prediction_key(reference_candidate.rows_by_participant[participant])
        participant_predictions: list[dict] = []
        for key, reference_row in reference_rows.items():
            combined = {int(label): 0.0 for label in class_labels}
            usable_weights: list[tuple[MetaCandidate, float]] = []
            total_weight = 0.0
            score_weight_sum = 0.0
            for candidate, weight, candidate_rows, selection_score, _train_outer_rows in keyed_rows:
                row = candidate_rows.get(key)
                if row is None:
                    continue
                scores = _row_score_vector(row, class_labels=class_labels)
                if scores is None:
                    continue
                for label in class_labels:
                    combined[int(label)] += weight * scores[int(label)]
                usable_weights.append((candidate, weight))
                total_weight += weight
                score_weight_sum += weight * selection_score
            if total_weight <= 0.0:
                raise ValueError(
                    "Weighted artifact meta-selection requires per-class scores for "
                    f"participant {participant}; regenerate source artifacts after score export."
                )
            combined = {label: value / total_weight for label, value in combined.items()}
            participant_predictions.append(
                _apply_weighted_scores_to_row(
                    reference_row,
                    selector_name=selector_name,
                    class_labels=class_labels,
                    scores=combined,
                    candidate_weights=usable_weights,
                    nested_selection_metric=nested_selection_metric,
                    selection_score=score_weight_sum / total_weight,
                )
            )

        selected_predictions.extend(participant_predictions)
        _dominant_score, _dominant_index, dominant_candidate, _dominant_rows, dominant_weight = max(
            weighted_candidates,
            key=lambda item: (item[4], -item[1]),
        )
        selection_rows.append(
            {
                "test_participant": participant,
                "artifact_ensemble": selector_name,
                "selected_artifact_candidate": dominant_candidate.name,
                "selection_metric": _nested_selection_metric_label(nested_selection_metric),
                "selection_metric_name": nested_selection_metric,
                "selection_metric_value": sum(score * weight for score, _index, _candidate, _rows, weight in weighted_candidates),
                "selection_balanced_accuracy": sum(
                    _metric_mean(rows, "balanced_accuracy") * weight
                    for _score, _index, _candidate, rows, weight in weighted_candidates
                ),
                "selection_accuracy": sum(
                    _metric_mean(rows, "accuracy") * weight
                    for _score, _index, _candidate, rows, weight in weighted_candidates
                ),
                "selection_top2_accuracy": sum(
                    _metric_mean(rows, "top2_accuracy") * weight
                    for _score, _index, _candidate, rows, weight in weighted_candidates
                ),
                "selection_top3_accuracy": sum(
                    _metric_mean(rows, "top3_accuracy") * weight
                    for _score, _index, _candidate, rows, weight in weighted_candidates
                ),
                "selection_mean_true_label_rank": sum(
                    _metric_mean(rows, "mean_true_label_rank") * weight
                    for _score, _index, _candidate, rows, weight in weighted_candidates
                ),
                "selection_n_subjects": len(weighted_candidates[0][3]) if weighted_candidates else 0,
                "candidate_artifact_count": len(scored_candidates),
                "dominant_artifact_candidate": dominant_candidate.name,
                "dominant_artifact_candidate_weight": dominant_weight,
                "weighted_artifact_candidate_weights": ";".join(
                    f"{candidate.name}:{weight:.6g}" for _score, _index, candidate, _rows, weight in weighted_candidates
                ),
            }
        )

    outer_rows = _outer_rows(selector_name, selected_predictions, n_classes=n_classes)
    summary = _weighted_summary_row(
        selector_name=selector_name,
        prediction_rows=selected_predictions,
        outer_rows=outer_rows,
        n_classes=n_classes,
        selection_metric=nested_selection_metric,
        selected_candidate_counts=_counts_text(str(row["selected_artifact_candidate"]) for row in selection_rows),
        n_candidates=len(candidates),
        weight_temperature=weight_temperature,
    )
    return {
        "predictions": selected_predictions,
        "outer": outer_rows,
        "selection": selection_rows,
        "group_summary": [summary],
    }


def nested_weighted_score_select_prediction_files(
    paths: Sequence[Path],
    *,
    selector_name: str = "cross_mode_weighted_score_selector",
    nested_selection_metric: str = "balanced_accuracy",
    n_classes: int | None = None,
    weight_temperature: float = 0.02,
) -> dict[str, list[dict]]:
    candidates, inferred_n_classes = load_meta_candidates(paths, n_classes=n_classes)
    return nested_weighted_score_select_candidates(
        candidates,
        selector_name=selector_name,
        nested_selection_metric=nested_selection_metric,
        n_classes=inferred_n_classes,
        weight_temperature=weight_temperature,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="Artifact-ensemble prediction CSV to include.")
    parser.add_argument("--input-glob", action="append", default=[], help="Glob for artifact-ensemble prediction CSVs to include.")
    parser.add_argument("--selector-name", default="cross_mode_weighted_score_selector")
    parser.add_argument(
        "--nested-selection-metric",
        choices=ARTIFACT_NESTED_SELECTION_METRIC_CHOICES,
        default="balanced_accuracy",
    )
    parser.add_argument("--weight-temperature", type=float, default=0.02)
    parser.add_argument("--n-classes", type=int, help="Override inferred class count.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="artifact_ensemble_cross_mode_weighted_score")
    args = parser.parse_args(argv)

    input_paths = _expand_input_paths(args.input, args.input_glob)
    artifacts = nested_weighted_score_select_prediction_files(
        input_paths,
        selector_name=args.selector_name,
        nested_selection_metric=args.nested_selection_metric,
        n_classes=args.n_classes,
        weight_temperature=args.weight_temperature,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output_dir / f"{args.output_stem}_predictions.csv", artifacts["predictions"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_outer.csv", artifacts["outer"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_selection.csv", artifacts["selection"])
    write_csv_rows(args.output_dir / f"{args.output_stem}_group_summary.csv", artifacts["group_summary"])
    write_markdown_summary(args.output_dir / f"{args.output_stem}_comparison.md", artifacts["group_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
