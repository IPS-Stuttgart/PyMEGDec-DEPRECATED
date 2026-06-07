from __future__ import annotations

from pymegdec.artifact_ensemble_meta_selector import MetaCandidate, _outer_rows, nested_meta_select_candidates
from pymegdec.artifact_ensemble_weighted_meta_selector import nested_weighted_score_select_candidates


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


def _scored_row(
    participant: int,
    trial_index: int,
    true_label: int,
    predicted_label: int,
    class_0_score: float,
    class_1_score: float,
) -> dict[str, str]:
    row = _row(participant, trial_index, true_label, predicted_label)
    row["score_class_0"] = f"{class_0_score:.6f}"
    row["score_class_1"] = f"{class_1_score:.6f}"
    row["score_1"] = f"{class_0_score:.6f}"
    row["score_2"] = f"{class_1_score:.6f}"
    return row


def _candidate(name: str, rows: list[dict[str, str]]) -> MetaCandidate:
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


def test_weighted_score_selector_can_beat_tie_broken_select_one() -> None:
    first_candidate = _candidate(
        "first_candidate",
        [
            _scored_row(1, 1, 0, 1, 0.49, 0.51),
            _scored_row(2, 1, 0, 0, 0.90, 0.10),
            _scored_row(3, 1, 1, 1, 0.10, 0.90),
            _scored_row(4, 1, 0, 0, 0.90, 0.10),
        ],
    )
    second_candidate = _candidate(
        "second_candidate",
        [
            _scored_row(1, 1, 0, 0, 0.55, 0.45),
            _scored_row(2, 1, 0, 0, 0.90, 0.10),
            _scored_row(3, 1, 1, 1, 0.10, 0.90),
            _scored_row(4, 1, 0, 0, 0.90, 0.10),
        ],
    )

    selected = nested_meta_select_candidates(
        [first_candidate, second_candidate],
        selector_name="select_one",
        nested_selection_metric="balanced_accuracy",
        n_classes=2,
    )
    selected_p1 = [row for row in selected["predictions"] if row["test_participant"] == "1"][0]
    assert selected_p1["predicted_label"] == "1"

    weighted = nested_weighted_score_select_candidates(
        [first_candidate, second_candidate],
        selector_name="weighted_score",
        nested_selection_metric="balanced_accuracy",
        n_classes=2,
    )
    weighted_p1 = [row for row in weighted["predictions"] if row["test_participant"] == "1"][0]

    assert weighted_p1["predicted_label"] == 0
    assert weighted_p1["artifact_ensemble_meta_selection"] == "leave_subject_out_weighted_score"
    assert weighted_p1["top2_correct"] is True
    assert float(weighted_p1["score_class_0"]) > float(weighted_p1["score_class_1"])
    assert weighted_p1["rank_class_0"] == 1
    assert weighted["group_summary"][0]["artifact_ensemble_meta_selection"] == "leave_subject_out_weighted_score"
