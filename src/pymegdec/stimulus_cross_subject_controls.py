"""Training-label controls for cross-subject stimulus benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from math import comb
from typing import Iterable

import numpy as np

from pymegdec import stimulus_cross_subject as base

LABEL_CONTROL_NONE = "none"
LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT = "shuffle-within-subject"
LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT = "circular-shift-within-subject"
LABEL_CONTROL_MODES = (
    LABEL_CONTROL_NONE,
    LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
    LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT,
)


def normalize_label_control(value: str | None) -> str:
    """Normalize a label-control token from CLI/workflow inputs."""

    normalized = (value or LABEL_CONTROL_NONE).strip().lower().replace("_", "-")
    aliases = {
        "": LABEL_CONTROL_NONE,
        "none": LABEL_CONTROL_NONE,
        "off": LABEL_CONTROL_NONE,
        "false": LABEL_CONTROL_NONE,
        "shuffle": LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
        "shuffle-within-subject": LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
        "within-subject-shuffle": LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT,
        "circular-shift": LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT,
        "circular-shift-within-subject": LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT,
        "within-subject-circular-shift": LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT,
    }
    if normalized not in aliases:
        raise ValueError(f"label_control must be one of {LABEL_CONTROL_MODES}.")
    return aliases[normalized]


def exact_one_sided_sign_p_value(n_above: int, n_total: int) -> float:
    """Exact one-sided binomial sign-test p-value for chance p=0.5."""

    n_total = int(n_total)
    n_above = int(n_above)
    if n_total <= 0:
        return float("nan")
    n_above = max(0, min(n_above, n_total))
    return float(sum(comb(n_total, k) for k in range(n_above, n_total + 1)) / (2**n_total))


def evaluate_cross_subject_stimulus_smoke_controlled(
    data_folder,
    participants,
    *,
    config=None,
    label_control: str = LABEL_CONTROL_NONE,
    label_control_seed: int | None = 0,
    progress=None,
):
    """Run the fixed cross-subject benchmark with optional training-label controls."""

    label_control = normalize_label_control(label_control)
    with training_label_control(label_control, label_control_seed):
        artifacts = base.evaluate_cross_subject_stimulus_smoke(data_folder, participants, config=config, progress=progress)
    return annotate_cross_subject_artifacts(artifacts, label_control=label_control, label_control_seed=label_control_seed)


def evaluate_nested_cross_subject_stimulus_controlled(
    data_folder,
    participants,
    *,
    candidate_configs,
    label_control: str = LABEL_CONTROL_NONE,
    label_control_seed: int | None = 0,
    outer_participants=None,
    progress=None,
    existing_artifacts=None,
    after_outer_fold=None,
):
    """Run nested cross-subject benchmark with optional training-label controls."""

    label_control = normalize_label_control(label_control)

    def _annotating_after_outer_fold(artifacts):
        if after_outer_fold is None:
            return None
        return after_outer_fold(
            annotate_cross_subject_artifacts(
                artifacts,
                label_control=label_control,
                label_control_seed=label_control_seed,
            )
        )

    with training_label_control(label_control, label_control_seed):
        artifacts = base.evaluate_nested_cross_subject_stimulus(
            data_folder,
            participants,
            candidate_configs=candidate_configs,
            outer_participants=outer_participants,
            progress=progress,
            existing_artifacts=existing_artifacts,
            after_outer_fold=_annotating_after_outer_fold,
        )
    return annotate_cross_subject_artifacts(artifacts, label_control=label_control, label_control_seed=label_control_seed)


@contextmanager
def training_label_control(label_control: str, label_control_seed: int | None):
    """Patch cross-subject fitting so only training labels are controlled."""

    label_control = normalize_label_control(label_control)
    if label_control == LABEL_CONTROL_NONE:
        yield
        return

    original_fit = base._fit_outer_fold_model  # pylint: disable=protected-access

    def _controlled_fit(train_sets, config, classifier_param, *, label_shuffle_seed=None, label_shuffle_context=()):
        controlled_sets = controlled_training_sets(train_sets, label_control=label_control, label_control_seed=label_control_seed)
        return original_fit(
            controlled_sets,
            config,
            classifier_param,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )

    base._fit_outer_fold_model = _controlled_fit  # pylint: disable=protected-access
    try:
        yield
    finally:
        base._fit_outer_fold_model = original_fit  # pylint: disable=protected-access


def controlled_training_sets(train_sets: Iterable, *, label_control: str, label_control_seed: int | None):
    """Return copies of participant feature sets with controlled training labels."""

    label_control = normalize_label_control(label_control)
    return [
        replace(
            feature_set,
            labels=controlled_labels(
                feature_set.labels,
                participant=feature_set.participant,
                label_control=label_control,
                label_control_seed=label_control_seed,
            ),
        )
        for feature_set in train_sets
    ]


def controlled_labels(labels, *, participant: int, label_control: str, label_control_seed: int | None):
    """Return controlled labels for one training participant."""

    labels = np.asarray(labels).copy()
    label_control = normalize_label_control(label_control)
    if label_control == LABEL_CONTROL_NONE:
        return labels
    rng = np.random.default_rng(_participant_seed(label_control_seed, participant))
    if label_control == LABEL_CONTROL_SHUFFLE_WITHIN_SUBJECT:
        return rng.permutation(labels)
    if label_control == LABEL_CONTROL_CIRCULAR_SHIFT_WITHIN_SUBJECT:
        if labels.size <= 1:
            return labels
        offset = int(rng.integers(1, labels.size))
        return np.roll(labels, offset)
    raise ValueError(f"Unsupported label_control: {label_control}")


def annotate_cross_subject_artifacts(artifacts: dict, *, label_control: str, label_control_seed: int | None):
    """Attach control metadata and exact sign-test p-values to benchmark artifacts."""

    label_control = normalize_label_control(label_control)
    annotated = {key: [dict(row) for row in rows] for key, rows in artifacts.items()}
    for key, rows in annotated.items():
        for row in rows:
            row["label_control"] = label_control
            row["label_control_seed"] = "" if label_control_seed is None else int(label_control_seed)
            row["training_labels_controlled"] = bool(label_control != LABEL_CONTROL_NONE)
            if key == "group_summary":
                _add_exact_sign_test_fields(row, annotated.get("outer", []))
    return annotated


def _add_exact_sign_test_fields(summary_row: dict, outer_rows: list[dict]) -> None:
    if not outer_rows:
        return
    chance = float(outer_rows[0]["chance_accuracy"])
    n_total = len(outer_rows)
    n_above = sum(float(row["balanced_accuracy"]) > chance for row in outer_rows)
    summary_row["participants_total"] = n_total
    summary_row["participants_above_chance"] = n_above
    summary_row["participants_at_or_below_chance"] = n_total - n_above
    summary_row["one_sided_exact_sign_p_value"] = exact_one_sided_sign_p_value(n_above, n_total)


def _participant_seed(seed: int | None, participant: int) -> int:
    base_seed = 0 if seed is None else int(seed)
    return int((base_seed * 1_000_003 + int(participant) * 9_176 + 0x9E3779B9) % (2**32))
