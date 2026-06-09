from __future__ import annotations

from pymegdec.artifact_ensemble_meta_selector import MetaCandidate, _outer_rows


def artifact_row(
    participant: int,
    trial_index: int,
    true_label: int,
    predicted_label: int,
) -> dict[str, str]:
    true_rank = 1.0 if true_label == predicted_label else 2.0
    return {
        "test_participant": str(participant),
        "test_trial_index": str(trial_index),
        "true_label": str(true_label),
        "predicted_label": str(predicted_label),
        "true_stimulus": str(true_label + 1),
        "predicted_stimulus": str(predicted_label + 1),
        "correct": str(true_label == predicted_label),
        "true_label_rank": str(true_rank),
        "top2_correct": "True",
        "top3_correct": "True",
        "artifact_ensemble": "candidate",
    }


def artifact_candidate_from_rows(
    name: str,
    rows: list[dict[str, str]],
    *,
    n_classes: int = 2,
) -> MetaCandidate:
    by_participant: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_participant.setdefault(row["test_participant"], []).append(row)
    outer_rows = _outer_rows(name, rows, n_classes=n_classes)
    return MetaCandidate(
        name=name,
        source_file=f"{name}.csv",
        original_ensemble=name,
        rows=rows,
        rows_by_participant=by_participant,
        outer_rows=outer_rows,
        outer_by_participant={str(row["test_participant"]): row for row in outer_rows},
    )


def artifact_candidate_from_correctness(
    name: str,
    correctness_by_participant: dict[int, bool],
) -> MetaCandidate:
    rows: list[dict[str, str]] = []
    for participant, correct in correctness_by_participant.items():
        if correct:
            rows.extend(
                [
                    artifact_row(participant, 1, 0, 0),
                    artifact_row(participant, 2, 1, 1),
                ]
            )
        else:
            rows.extend(
                [
                    artifact_row(participant, 1, 0, 1),
                    artifact_row(participant, 2, 1, 0),
                ]
            )
    return artifact_candidate_from_rows(name, rows)
