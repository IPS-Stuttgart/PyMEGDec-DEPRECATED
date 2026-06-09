from __future__ import annotations

from pymegdec.artifact_ensemble_meta_selector import nested_meta_select_candidates
from tests.artifact_ensemble_meta_selector_helpers import artifact_candidate_from_correctness


def test_nested_meta_selector_uses_other_subjects_only() -> None:
    overfit_to_participant_1 = artifact_candidate_from_correctness(
        "overfit_to_participant_1",
        {1: True, 2: False, 3: False, 4: False},
    )
    broadly_good = artifact_candidate_from_correctness(
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
