"""Leakage-safe meta-selection across artifact-ensemble prediction outputs.

This module consumes prediction CSVs produced by ``pymegdec.stimulus_artifact_ensemble``
for many aggregation modes/normalizations and adds one more leave-subject-out
selection layer.  For each held-out participant it chooses the candidate whose
outer-fold performance is best on the other participants, then copies only that
candidate's predictions for the held-out participant.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.stimulus_artifact_ensemble import ARTIFACT_NESTED_SELECTION_METRIC_CHOICES

CLASS_SCORE_RE = re.compile(r"^(?:score|prob)_class_(-?\d+)$")
STIMULUS_SCORE_RE = re.compile(r"^(?:score|prob)_([1-9]\d*)$")
CLASS_RANK_RE = re.compile(r"^rank_class_(-?\d+)$")
STIMULUS_RANK_RE = re.compile(r"^rank_([1-9]\d*)$")
PREDICTION_GLOB = "*_predictions.csv"


@dataclass(frozen=True)
class MetaCandidate:
    """One candidate artifact ensemble loaded from a prediction CSV."""

    name: str
    source_file: str
    original_ensemble: str
    rows: list[dict[str, str]]
    rows_by_participant: dict[str, list[dict[str, str]]]
    outer_rows: list[dict[str, object]]
    outer_by_participant: dict[str, dict[str, object]]


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


def _to_int(value: object, *, field: str) -> int:
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Missing integer value for {field}.")
    return int(float(text))


def _to_float(value: object, *, default: float = math.nan) -> float:
    text = str(value).strip()
    if text == "":
        return default
    try:
        value_float = float(text)
    except ValueError:
        return default
    return value_float if math.isfinite(value_float) else default


def _parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(value)


def _participant_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):012d}") if text.isdigit() else (1, text)


def _metric_mean(rows: Sequence[dict[str, object]], metric: str) -> float:
    return _mean([_to_float(row.get(metric, "")) for row in rows])


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


def _format_percent(value: float) -> str:
    return "" if not math.isfinite(value) else f"{100.0 * value:.6f}"


def _label_from_row(row: Mapping[str, object], *, label_column: str, stimulus_column: str, field: str) -> int:
    raw_label = row.get(label_column)
    if raw_label is not None and str(raw_label).strip() != "":
        return _to_int(raw_label, field=field)
    raw_stimulus = row.get(stimulus_column)
    if raw_stimulus is None or str(raw_stimulus).strip() == "":
        raise ValueError(f"Missing {label_column} or {stimulus_column}.")
    return _to_int(raw_stimulus, field=stimulus_column) - 1


def _row_correct(row: Mapping[str, object]) -> bool:
    if "correct" in row and str(row.get("correct", "")).strip() != "":
        return _parse_bool(row["correct"])
    return _label_from_row(row, label_column="true_label", stimulus_column="true_stimulus", field="true_label") == _label_from_row(
        row,
        label_column="predicted_label",
        stimulus_column="predicted_stimulus",
        field="predicted_label",
    )


def _row_true_rank(row: Mapping[str, object], *, n_classes: int) -> float:
    rank = _to_float(row.get("true_label_rank", ""))
    if math.isfinite(rank):
        return rank
    return 1.0 if _row_correct(row) else float(max(n_classes, 1))


def _row_topk_correct(row: Mapping[str, object], *, column: str, k: int, n_classes: int) -> bool:
    if column in row and str(row.get(column, "")).strip() != "":
        return _parse_bool(row[column])
    return _row_true_rank(row, n_classes=n_classes) <= k


def _balanced_accuracy(rows: Sequence[Mapping[str, object]]) -> float:
    by_label: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        by_label[_label_from_row(row, label_column="true_label", stimulus_column="true_stimulus", field="true_label")].append(_row_correct(row))
    recalls = [sum(values) / len(values) for values in by_label.values() if values]
    return sum(recalls) / len(recalls) if recalls else math.nan


def _class_labels_from_rows(rows: Iterable[Mapping[str, object]]) -> list[int]:
    labels: set[int] = set()
    for row in rows:
        for label_column, stimulus_column, field in (
            ("true_label", "true_stimulus", "true_label"),
            ("predicted_label", "predicted_stimulus", "predicted_label"),
        ):
            try:
                labels.add(_label_from_row(row, label_column=label_column, stimulus_column=stimulus_column, field=field))
            except ValueError:
                pass
        for column, value in row.items():
            if str(value).strip() == "":
                continue
            class_match = CLASS_SCORE_RE.match(column) or CLASS_RANK_RE.match(column)
            if class_match:
                labels.add(int(class_match.group(1)))
                continue
            stimulus_match = STIMULUS_SCORE_RE.match(column) or STIMULUS_RANK_RE.match(column)
            if stimulus_match:
                labels.add(int(stimulus_match.group(1)) - 1)
    return sorted(labels)


def _outer_rows(candidate_name: str, prediction_rows: Sequence[Mapping[str, object]], *, n_classes: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_participant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        participant = str(row.get("test_participant", ""))
        if not participant:
            raise ValueError(f"Candidate {candidate_name!r} has a row without test_participant.")
        by_participant[participant].append(row)
    for participant, participant_rows in sorted(by_participant.items(), key=lambda item: _participant_sort_key(item[0])):
        n_test = len(participant_rows)
        correct = [_row_correct(row) for row in participant_rows]
        ranks = [_row_true_rank(row, n_classes=n_classes) for row in participant_rows]
        rows.append(
            {
                "artifact_candidate": candidate_name,
                "test_participant": participant,
                "accuracy": sum(correct) / n_test if n_test else math.nan,
                "balanced_accuracy": _balanced_accuracy(participant_rows),
                "chance_accuracy": 1.0 / n_classes if n_classes else math.nan,
                "top2_accuracy": sum(_row_topk_correct(row, column="top2_correct", k=2, n_classes=n_classes) for row in participant_rows) / n_test if n_test else math.nan,
                "top3_accuracy": sum(_row_topk_correct(row, column="top3_correct", k=3, n_classes=n_classes) for row in participant_rows) / n_test if n_test else math.nan,
                "mean_true_label_rank": _mean(ranks),
                "n_test": n_test,
            }
        )
    return rows


def _artifact_recipe_rank_score(row: dict[str, object], *, n_classes: int) -> float:
    if n_classes <= 0:
        raise ValueError("Nested artifact meta-selection requires at least one class.")
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


def _normalize_nested_selection_metric(selection_metric: str) -> str:
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
            "Artifact meta-selection metric must be one of "
            f"{', '.join(ARTIFACT_NESTED_SELECTION_METRIC_CHOICES)}."
        )
    return normalized


def _nested_selection_metric_label(selection_metric: str) -> str:
    return f"other_subjects_{_normalize_nested_selection_metric(selection_metric)}"


def _nested_selection_metric_value(rows: Sequence[dict[str, object]], *, selection_metric: str, n_classes: int) -> float:
    normalized = _normalize_nested_selection_metric(selection_metric)
    if normalized == "balanced_accuracy":
        return _metric_mean(rows, "balanced_accuracy")
    if normalized == "balanced_accuracy_lcb":
        values = [_to_float(row["balanced_accuracy"]) for row in rows]
        return _mean(values) - _sem(values)
    values = [_artifact_recipe_rank_score(row, n_classes=n_classes) for row in rows]
    score = _mean(values)
    if normalized.endswith("_lcb"):
        score -= _sem(values)
    return score


def _counts_text(values: Iterable[str]) -> str:
    counts = Counter(values)
    return ";".join(f"{value}:{counts[value]}" for value in sorted(counts, key=_participant_sort_key))


def _summary_row(
    *,
    selector_name: str,
    prediction_rows: Sequence[dict[str, object]],
    outer_rows: Sequence[dict[str, object]],
    n_classes: int,
    selection_metric: str,
    selected_candidate_counts: str,
    n_candidates: int,
) -> dict[str, object]:
    accuracies = [_to_float(row["accuracy"]) for row in outer_rows]
    balanced = [_to_float(row["balanced_accuracy"]) for row in outer_rows]
    top2 = [_to_float(row["top2_accuracy"]) for row in outer_rows]
    top3 = [_to_float(row["top3_accuracy"]) for row in outer_rows]
    ranks = [_to_float(row["mean_true_label_rank"]) for row in outer_rows]
    chance = 1.0 / n_classes if n_classes else math.nan
    return {
        "artifact_ensemble": selector_name,
        "artifact_ensemble_meta_selection": "leave_subject_out_cross_candidate",
        "selection_metric": _nested_selection_metric_label(selection_metric),
        "selection_metric_name": _normalize_nested_selection_metric(selection_metric),
        "candidate_artifact_count": n_candidates,
        "selected_artifact_candidate_counts": selected_candidate_counts,
        "n_outer_folds": len(outer_rows),
        "n_predictions": len(prediction_rows),
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


def _candidate_name(path: Path, ensemble_name: str) -> str:
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    return f"{stem}::{ensemble_name or 'unnamed'}"


def load_meta_candidates(paths: Sequence[Path], *, n_classes: int | None = None) -> tuple[list[MetaCandidate], int]:
    grouped_rows: list[tuple[str, str, str, list[dict[str, str]]]] = []
    all_rows: list[dict[str, str]] = []
    for path in paths:
        rows = read_csv_rows(path)
        if not rows:
            continue
        all_rows.extend(rows)
        by_ensemble: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_ensemble[str(row.get("artifact_ensemble", "")).strip() or "unnamed"].append(row)
        for ensemble_name, ensemble_rows in by_ensemble.items():
            grouped_rows.append((_candidate_name(path, ensemble_name), path.name, ensemble_name, ensemble_rows))
    if not grouped_rows:
        raise ValueError("No artifact ensemble prediction rows were found.")

    inferred_n_classes = int(n_classes) if n_classes is not None else len(_class_labels_from_rows(all_rows))
    if inferred_n_classes <= 0:
        raise ValueError("Could not infer the number of classes from prediction rows.")

    seen_names: set[str] = set()
    candidates: list[MetaCandidate] = []
    for name, source_file, original_ensemble, rows in grouped_rows:
        if name in seen_names:
            raise ValueError(f"Duplicate artifact candidate name {name!r}; input file stems must be unique.")
        seen_names.add(name)
        by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            participant = str(row.get("test_participant", ""))
            if not participant:
                raise ValueError(f"Candidate {name!r} contains a row without test_participant.")
            by_participant[participant].append(row)
        outer_rows = _outer_rows(name, rows, n_classes=inferred_n_classes)
        candidates.append(
            MetaCandidate(
                name=name,
                source_file=source_file,
                original_ensemble=original_ensemble,
                rows=rows,
                rows_by_participant=dict(by_participant),
                outer_rows=outer_rows,
                outer_by_participant={str(row["test_participant"]): row for row in outer_rows},
            )
        )
    return candidates, inferred_n_classes


def nested_meta_select_candidates(
    candidates: Sequence[MetaCandidate],
    *,
    selector_name: str = "cross_mode_nested_selector",
    nested_selection_metric: str = "balanced_accuracy",
    n_classes: int,
) -> dict[str, list[dict]]:
    """Select one artifact candidate per held-out subject using other subjects only."""

    nested_selection_metric = _normalize_nested_selection_metric(nested_selection_metric)
    participants = sorted(
        set().union(*(set(candidate.outer_by_participant) for candidate in candidates)),
        key=_participant_sort_key,
    )
    if not participants:
        raise ValueError("Nested artifact meta-selection requires test_participant values.")

    selected_predictions: list[dict] = []
    selection_rows: list[dict] = []
    for participant in participants:
        scored_candidates: list[tuple[float, float, int, MetaCandidate, list[dict[str, object]]]] = []
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
            balanced = _metric_mean(train_outer_rows, "balanced_accuracy")
            scored_candidates.append((selection_score, balanced, -candidate_index, candidate, train_outer_rows))
        if not scored_candidates:
            raise ValueError(f"Cannot select an artifact candidate for participant {participant}; no candidate has source-subject rows.")

        selected_score, selected_balanced, _tie_breaker, selected_candidate, train_outer_rows = max(scored_candidates)
        selection_rows.append(
            {
                "test_participant": participant,
                "artifact_ensemble": selector_name,
                "selected_artifact_candidate": selected_candidate.name,
                "selected_artifact_candidate_source_file": selected_candidate.source_file,
                "selected_artifact_candidate_original_ensemble": selected_candidate.original_ensemble,
                "selection_metric": _nested_selection_metric_label(nested_selection_metric),
                "selection_metric_name": nested_selection_metric,
                "selection_metric_value": selected_score,
                "selection_balanced_accuracy": selected_balanced,
                "selection_accuracy": _metric_mean(train_outer_rows, "accuracy"),
                "selection_top2_accuracy": _metric_mean(train_outer_rows, "top2_accuracy"),
                "selection_top3_accuracy": _metric_mean(train_outer_rows, "top3_accuracy"),
                "selection_mean_true_label_rank": _metric_mean(train_outer_rows, "mean_true_label_rank"),
                "selection_n_subjects": len(train_outer_rows),
                "candidate_artifact_count": len(candidates),
            }
        )
        for row in selected_candidate.rows_by_participant[participant]:
            selected_row: dict[str, object] = dict(row)
            selected_row["source_artifact_ensemble"] = row.get("artifact_ensemble", "")
            selected_row["artifact_ensemble"] = selector_name
            selected_row["artifact_ensemble_meta_selection"] = "leave_subject_out_cross_candidate"
            selected_row["selected_artifact_candidate"] = selected_candidate.name
            selected_row["selected_artifact_candidate_source_file"] = selected_candidate.source_file
            selected_row["selected_artifact_candidate_original_ensemble"] = selected_candidate.original_ensemble
            selected_row["selection_metric"] = _nested_selection_metric_label(nested_selection_metric)
            selected_row["selection_metric_name"] = nested_selection_metric
            selected_row["selection_metric_value"] = selected_score
            selected_row["selection_balanced_accuracy"] = selected_balanced
            selected_predictions.append(selected_row)

    outer_rows = _outer_rows(selector_name, selected_predictions, n_classes=n_classes)
    summary = _summary_row(
        selector_name=selector_name,
        prediction_rows=selected_predictions,
        outer_rows=outer_rows,
        n_classes=n_classes,
        selection_metric=nested_selection_metric,
        selected_candidate_counts=_counts_text(str(row["selected_artifact_candidate"]) for row in selection_rows),
        n_candidates=len(candidates),
    )
    return {
        "predictions": selected_predictions,
        "outer": outer_rows,
        "selection": selection_rows,
        "group_summary": [summary],
    }


def nested_meta_select_prediction_files(
    paths: Sequence[Path],
    *,
    selector_name: str = "cross_mode_nested_selector",
    nested_selection_metric: str = "balanced_accuracy",
    n_classes: int | None = None,
) -> dict[str, list[dict]]:
    candidates, inferred_n_classes = load_meta_candidates(paths, n_classes=n_classes)
    return nested_meta_select_candidates(
        candidates,
        selector_name=selector_name,
        nested_selection_metric=nested_selection_metric,
        n_classes=inferred_n_classes,
    )


def _expand_input_paths(paths: Sequence[str], globs: Sequence[str]) -> list[Path]:
    expanded: list[Path] = [Path(path) for path in paths]
    for pattern in globs:
        expanded.extend(sorted(Path().glob(pattern)))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in expanded:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise ValueError("No input prediction CSVs matched.")
    missing = [path for path in unique if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input prediction CSV(s) not found: {', '.join(str(path) for path in missing)}")
    return unique


def write_markdown_summary(path: Path, summary_rows: Sequence[dict]) -> None:
    lines = [
        "# Cross-Mode Artifact Meta-Selector",
        "",
        "| selector | metric | candidates | folds | balanced | accuracy | top-2 | top-3 | mean rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {metric} | {candidates} | {folds} | {balanced:.2f}% | {accuracy:.2f}% | {top2:.2f}% | {top3:.2f}% | {rank:.3f} |".format(
                name=row["artifact_ensemble"],
                metric=row.get("selection_metric_name", ""),
                candidates=row.get("candidate_artifact_count", ""),
                folds=row.get("n_outer_folds", ""),
                balanced=100.0 * _to_float(row.get("balanced_accuracy_mean", "")),
                accuracy=100.0 * _to_float(row.get("accuracy_mean", "")),
                top2=100.0 * _to_float(row.get("top2_accuracy_mean", "")),
                top3=100.0 * _to_float(row.get("top3_accuracy_mean", "")),
                rank=_to_float(row.get("mean_true_label_rank_mean", "")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="Artifact-ensemble prediction CSV to include.")
    parser.add_argument("--input-glob", action="append", default=[], help="Glob for artifact-ensemble prediction CSVs to include.")
    parser.add_argument("--selector-name", default="cross_mode_nested_selector")
    parser.add_argument(
        "--nested-selection-metric",
        choices=ARTIFACT_NESTED_SELECTION_METRIC_CHOICES,
        default="balanced_accuracy",
    )
    parser.add_argument("--n-classes", type=int, help="Override inferred class count.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="artifact_ensemble_cross_mode_nested")
    args = parser.parse_args(argv)

    input_paths = _expand_input_paths(args.input, args.input_glob)
    artifacts = nested_meta_select_prediction_files(
        input_paths,
        selector_name=args.selector_name,
        nested_selection_metric=args.nested_selection_metric,
        n_classes=args.n_classes,
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
