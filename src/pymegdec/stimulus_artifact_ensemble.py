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
    source_names = tuple(token.strip() for token in raw_sources.split(",") if token.strip())
    if not name.strip() or not source_names:
        raise ValueError(f"Invalid ensemble specification: {spec!r}.")
    return name.strip(), source_names


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
) -> list[int]:
    predictions = [
        _label_from_row(row, label_column="predicted_label", stimulus_column="predicted_stimulus", field="predicted_label")
        for row in source_rows
    ]
    votes = Counter(predictions)
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


def _class_value_columns(rows: Sequence[dict[str, str]], pattern: re.Pattern[str]) -> dict[int, str]:
    common: dict[int, str] | None = None
    for row in rows:
        columns = {
            int(match.group(1)): column
            for column in row
            if (match := pattern.match(column)) and str(row.get(column, "")).strip() != ""
        }
        common = columns if common is None else {label: column for label, column in common.items() if label in columns}
    return common or {}


def _rank_labels_by_scores(source_rows: Sequence[dict[str, str]], *, class_labels: Sequence[int]) -> tuple[list[int], str] | None:
    score_columns = _class_value_columns(source_rows, CLASS_SCORE_RE)
    if score_columns:
        scores: dict[int, float] = {}
        for label in class_labels:
            column = score_columns.get(label)
            if column is None:
                continue
            values = [float(row[column]) for row in source_rows]
            scores[label] = sum(values) / len(values)
        if scores:
            return sorted(class_labels, key=lambda label: (-scores.get(label, float("-inf")), label)), "class_score_mean"

    rank_columns = _class_value_columns(source_rows, CLASS_RANK_RE)
    if rank_columns:
        borda_scores: dict[int, float] = {}
        for label in class_labels:
            column = rank_columns.get(label)
            if column is None:
                continue
            values = [float(row[column]) for row in source_rows]
            borda_scores[label] = -sum(values) / len(values)
        if borda_scores:
            return sorted(class_labels, key=lambda label: (-borda_scores.get(label, float("-inf")), label)), "class_rank_borda"

    return None


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
    source_rows: Sequence[dict[str, str]],
    key_columns: Sequence[str],
    key: tuple[str, ...],
    class_labels: Sequence[int],
) -> dict[str, object]:
    reference = source_rows[0]
    true_label = _label_from_row(reference, label_column="true_label", stimulus_column="true_stimulus", field="true_label")
    source_predictions = [
        _label_from_row(row, label_column="predicted_label", stimulus_column="predicted_stimulus", field="predicted_label")
        for row in source_rows
    ]
    if len(source_rows) == 1:
        ranked = _rank_labels_by_scores(source_rows, class_labels=class_labels)
        if ranked is not None:
            ranked_labels, rank_source = ranked
            true_rank = float(ranked_labels.index(true_label) + 1)
        else:
            ranked_labels = [source_predictions[0], *[label for label in class_labels if label != source_predictions[0]]]
            source_true_rank = _source_rank(reference)
            true_rank = source_true_rank if source_true_rank is not None else float(ranked_labels.index(true_label) + 1)
            rank_source = "source_true_label_rank" if source_true_rank is not None else "hard_vote"
    else:
        ranked = _rank_labels_by_scores(source_rows, class_labels=class_labels)
        if ranked is None:
            ranked_labels = _rank_labels_by_hard_votes(source_rows, class_labels=class_labels)
            rank_source = "hard_vote"
        else:
            ranked_labels, rank_source = ranked
        true_rank = float(ranked_labels.index(true_label) + 1)
    predicted_label = int(ranked_labels[0])
    row: dict[str, object] = {
        "artifact_ensemble": ensemble_name,
        "artifact_ensemble_sources": ";".join(source_names),
        "artifact_ensemble_source_count": len(source_names),
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


def _group_summary(ensemble_name: str, source_names: Sequence[str], outer_rows: Sequence[dict[str, object]], *, n_classes: int) -> dict[str, object]:
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
    summary = _group_summary(selector_name, ensemble_order, outer_rows, n_classes=n_classes)
    summary["artifact_ensemble_recipe_selection"] = "leave_subject_out"
    summary["selection_metric"] = "other_subjects_balanced_accuracy"
    summary["selected_artifact_ensemble_counts"] = _counts_text(
        str(row["selected_artifact_ensemble"]) for row in selection_rows
    )
    return selected_predictions, outer_rows, selection_rows, summary


def ensemble_prediction_sources(
    sources: Sequence[PredictionSource],
    ensembles: Sequence[tuple[str, Sequence[str]]],
    *,
    key_columns: Sequence[str] = DEFAULT_KEY_COLUMNS,
    nested_selector_name: str | None = None,
) -> dict[str, list[dict]]:
    """Build hard-vote artifact ensembles from already completed prediction CSVs."""

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
    for ensemble_name, source_names in ensembles:
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
                source_rows=[indexed_sources[source_name][key] for source_name in source_names],
                key_columns=key_columns,
                key=key,
                class_labels=class_labels,
            )
            for key in sorted(reference_keys, key=lambda values: tuple(int(value) if str(value).isdigit() else value for value in values))
        ]
        outer_rows = _outer_rows(ensemble_name, prediction_rows, n_classes=len(class_labels))
        summary = _group_summary(ensemble_name, source_names, outer_rows, n_classes=len(class_labels))
        prediction_rows_by_ensemble[ensemble_name] = prediction_rows
        outer_rows_by_ensemble[ensemble_name] = outer_rows
        all_predictions.extend(prediction_rows)
        all_outer.extend(outer_rows)
        all_summary.append(summary)
    artifacts: dict[str, list[dict]] = {"predictions": all_predictions, "outer": all_outer, "group_summary": all_summary}
    if nested_selector_name:
        nested_predictions, nested_outer, nested_selection, nested_summary = _nested_subject_selector(
            selector_name=nested_selector_name,
            ensemble_order=[name for name, _ in ensembles],
            ensemble_sources=ensemble_sources,
            prediction_rows_by_ensemble=prediction_rows_by_ensemble,
            outer_rows_by_ensemble=outer_rows_by_ensemble,
            n_classes=len(class_labels),
        )
        artifacts["predictions"].extend(nested_predictions)
        artifacts["outer"].extend(nested_outer)
        artifacts["group_summary"].append(nested_summary)
        artifacts["nested_selection"] = nested_selection
    return artifacts


def write_markdown_summary(path: Path, summary_rows: Sequence[dict]) -> None:
    lines = [
        "# Artifact Prediction Ensembles",
        "",
        "| ensemble | sources | folds | balanced | accuracy | top-2 | top-3 | mean rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {name} | {sources} | {folds} | {balanced:.2f}% | {accuracy:.2f}% | {top2:.2f}% | {top3:.2f}% | {rank:.3f} |".format(
                name=row["artifact_ensemble"],
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
    parser.add_argument("--ensemble", action="append", required=True, help="Named ensemble as name=source_a,source_b.")
    parser.add_argument("--key-column", action="append", dest="key_columns", help="Prediction-row key column. Defaults to test_participant, test_trial_index, true_label.")
    parser.add_argument(
        "--nested-selector-name",
        help="Optional leakage-safe leave-subject-out artifact recipe selector row to add to the outputs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="artifact_ensemble")
    args = parser.parse_args(argv)

    sources = [load_prediction_source(spec) for spec in args.source]
    ensembles = [parse_ensemble_spec(spec) for spec in args.ensemble]
    artifacts = ensemble_prediction_sources(
        sources,
        ensembles,
        key_columns=tuple(args.key_columns or DEFAULT_KEY_COLUMNS),
        nested_selector_name=args.nested_selector_name,
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
