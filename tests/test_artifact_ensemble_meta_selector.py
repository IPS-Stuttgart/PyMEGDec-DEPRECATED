from __future__ import annotations

from pymegdec.artifact_ensemble_meta_selector import MetaCandidate, _outer_rows, nested_meta_select_candidates


def _row(participant: int, trial_index: int, true_label: int, predicted_label: int) -> dict[str, str]:
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


def _participant_rows(participant: int, *, correct: bool) -> list[dict[str, str]]:
    if correct:
        return [
            _row(participant, 1, 0, 0),
            _row(participant, 2, 1, 1),
        ]
    return [
        _row(participant, 1, 0, 1),
        _row(participant, 2, 1, 0),
    ]


def _candidate(name: str, correctness_by_participant: dict[int, bool]) -> MetaCandidate:
    rows: list[dict[str, str]] = []
    for participant, correct in correctness_by_participant.items():
        rows.extend(_participant_rows(participant, correct=correct))
    by_participant: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_participant.setdefault(row["test_participant"], []).append(row)
    outer_rows = _outer_rows(name, rows, n_classes=2)
    return MetaCandidate(
        name=name,
        source_file=f"{name}.csv",
        original_ensemble=name,
        rows=rows,
        rows_by_participant=by_participant,
        outer_rows=outer_rows,
        outer_by_participant={str(row["test_participant"]): row for row in outer_rows},
    )


def test_nested_meta_selector_uses_other_subjects_only() -> None:
    overfit_to_participant_1 = _candidate(
        "overfit_to_participant_1",
        {1: True, 2: False, 3: False, 4: False},
    )
    broadly_good = _candidate(
        "broadly_good",
        {1: False, 2: True, 3: True, 4: True},
    )

    artifacts = nested_meta_select_candidates(
        [overfit_to_participant_1, broadly_good],
        selector_name="cross_mode_nested_selector",
        nested_selection_metric="balanced_accuracy",
        n_classes=2,
    )

    selection_by_participant = {
        row["test_participant"]: row["selected_artifact_candidate"]
        for row in artifacts["selection"]
    }

    assert selection_by_participant["1"] == "broadly_good"
    assert {row["artifact_ensemble"] for row in artifacts["predictions"]} == {"cross_mode_nested_selector"}
    assert artifacts["group_summary"][0]["candidate_artifact_count"] == 2
    assert artifacts["group_summary"][0]["selected_artifact_candidate_counts"] == "broadly_good:4"
