from unittest.mock import patch

import numpy as np

from pymegdec.stimulus_hyperalignment import CrossSubjectHyperalignmentConfig, evaluate_cross_subject_hyperalignment
from pymegdec.stimulus_mcca import CrossSubjectMCCAConfig, evaluate_cross_subject_mcca
from tests.matlab_fixtures import cell_array


def _mat_data(labels, values):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.asarray(labels, dtype=int)
    time = np.asarray([-0.5, 0.0, 0.1, 0.15, 0.2, 1.5], dtype=float)
    trials = []
    for label, value in zip(labels, values, strict=True):
        signal = np.zeros((2, time.size), dtype=float)
        signal[:, (time >= 0.15) & (time <= 0.25)] = value
        signal[:, (time >= -0.5) & (time <= 0.0)] = 0.1 * label
        trials.append(signal)
    return {"trial": cell_array(trials), "time": cell_array([time for _ in trials]), "trialinfo": trialinfo}


def _loadmat_side_effect(data_by_participant):
    def loadmat(path):
        stem = path.name if hasattr(path, "name") else str(path).rsplit("/", maxsplit=1)[-1]
        participant = int(stem.removeprefix("Part").removesuffix("Data.mat"))
        return {"data": np.array([data_by_participant[participant]], dtype=object)}

    return loadmat


def _toy_data_by_participant():
    return {
        1: _mat_data([1, 2, 1, 2], [-1.20, 1.20, -1.10, 1.10]),
        2: _mat_data([1, 2, 1, 2], [-1.00, 1.00, -0.90, 0.90]),
        3: _mat_data([1, 2, 1, 2], [-1.30, 1.30, -1.20, 1.20]),
        4: _mat_data([1, 2, 1, 2], [-1.10, 1.10, -1.00, 1.00]),
    }


def test_cross_subject_mcca_exports_full_loso_artifacts():
    config = CrossSubjectMCCAConfig(
        window_center=0.2,
        window_size=0.1,
        feature_mode="sensor_mean",
        normalization="none",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        mcca_components=2,
        mcca_sample_mode="class_repetition",
        chance_classes=2,
        signflip_permutations=32,
    )
    with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(_toy_data_by_participant())):
        artifacts = evaluate_cross_subject_mcca("unused", [1, 2, 3, 4], config=config)
    assert len(artifacts["outer"]) == 4
    assert len(artifacts["predictions"]) == 16
    assert len(artifacts["group_summary"]) == 1
    assert artifacts["confusion"]
    assert artifacts["per_stimulus"]
    assert all(row["alignment"] == "mcca_group_projection" for row in artifacts["outer"])
    assert all("top2_accuracy" in row and "mean_true_label_rank" in row for row in artifacts["outer"])
    assert all("true_label_rank" in row for row in artifacts["predictions"])


def test_cross_subject_mcca_target_calibration_excludes_calibration_trials():
    config = CrossSubjectMCCAConfig(
        window_center=0.2,
        window_size=0.1,
        feature_mode="sensor_mean",
        normalization="none",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        mcca_components=2,
        mcca_sample_mode="class_repetition",
        target_calibration_trials_per_class=1,
        chance_classes=2,
        signflip_permutations=32,
    )
    with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(_toy_data_by_participant())):
        artifacts = evaluate_cross_subject_mcca("unused", [1, 2, 3, 4], config=config)

    assert len(artifacts["outer"]) == 4
    assert len(artifacts["predictions"]) == 8
    assert len(artifacts["group_summary"]) == 1
    assert all(row["alignment"] == "mcca_target_calibrated" for row in artifacts["outer"])
    assert all(row["target_transform"] == "target_calibrated" for row in artifacts["outer"])
    assert all(row["n_target_calibration_trials"] == 2 for row in artifacts["outer"])
    assert all(row["n_scored_trials"] == 2 for row in artifacts["outer"])
    assert all(row["trial_index"] in {2, 3} for row in artifacts["predictions"])
    assert all(row["alignment"] == "mcca_target_calibrated" for row in artifacts["predictions"])
    assert all(row["target_transform"] == "target_calibrated" for row in artifacts["predictions"])
    assert "target_calibration_trials_per_class" in artifacts["group_summary"][0]


def test_cross_subject_mcca_can_use_separate_alignment_window():
    config = CrossSubjectMCCAConfig(
        window_center=0.2,
        window_size=0.1,
        alignment_window_center=0.05,
        alignment_window_size=0.3,
        feature_mode="sensor_flat",
        normalization="none",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        mcca_components=2,
        mcca_sample_mode="class_repetition",
        chance_classes=2,
        signflip_permutations=32,
    )
    with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(_toy_data_by_participant())):
        artifacts = evaluate_cross_subject_mcca("unused", [1, 2, 3, 4], config=config)
    assert len(artifacts["outer"]) == 4
    first_outer = artifacts["outer"][0]
    assert first_outer["window_center_s"] == 0.2
    assert first_outer["alignment_window_center_s"] == 0.05
    assert first_outer["window_size_s"] == 0.1
    assert first_outer["alignment_window_size_s"] == 0.3
    assert all(row["alignment_window_center_s"] == 0.05 for row in artifacts["predictions"])


def test_cross_subject_hyperalignment_exports_full_loso_artifacts():
    config = CrossSubjectHyperalignmentConfig(
        window_center=0.2,
        window_size=0.1,
        feature_mode="sensor_mean",
        normalization="none",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        hyperalignment_components=2,
        hyperalignment_sample_mode="class_repetition",
        chance_classes=2,
        signflip_permutations=32,
    )
    with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(_toy_data_by_participant())):
        artifacts = evaluate_cross_subject_hyperalignment("unused", [1, 2, 3, 4], config=config)
    assert len(artifacts["outer"]) == 4
    assert len(artifacts["predictions"]) == 16
    assert len(artifacts["group_summary"]) == 1
    assert artifacts["confusion"]
    assert artifacts["per_stimulus"]
    assert all(row["alignment"] == "class_hyperalignment_group_average" for row in artifacts["outer"])
    assert all("top2_accuracy" in row and "top3_accuracy" in row for row in artifacts["outer"])
    assert all("mean_true_label_rank" in row for row in artifacts["outer"])
    assert all("true_label_rank" in row and "score_class_1" in row for row in artifacts["predictions"])
    assert "top2_accuracy_mean" in artifacts["group_summary"][0]
    assert "mean_true_label_rank_mean" in artifacts["group_summary"][0]


def test_cross_subject_hyperalignment_can_use_separate_alignment_window():
    config = CrossSubjectHyperalignmentConfig(
        window_center=0.2,
        window_size=0.1,
        alignment_window_center=0.05,
        alignment_window_size=0.3,
        feature_mode="sensor_flat",
        normalization="none",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        hyperalignment_components=2,
        hyperalignment_sample_mode="class_repetition",
        chance_classes=2,
        signflip_permutations=32,
    )
    with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(_toy_data_by_participant())):
        artifacts = evaluate_cross_subject_hyperalignment("unused", [1, 2, 3, 4], config=config)
    assert len(artifacts["outer"]) == 4
    first_outer = artifacts["outer"][0]
    assert first_outer["window_center_s"] == 0.2
    assert first_outer["alignment_window_center_s"] == 0.05
    assert first_outer["window_size_s"] == 0.1
    assert first_outer["alignment_window_size_s"] == 0.3
    assert all(row["alignment_window_center_s"] == 0.05 for row in artifacts["predictions"])


def test_cross_subject_hyperalignment_label_shuffle_is_reproducible():
    config = CrossSubjectHyperalignmentConfig(
        window_center=0.2,
        window_size=0.1,
        feature_mode="sensor_mean",
        normalization="none",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        hyperalignment_components=2,
        chance_classes=2,
        signflip_permutations=0,
    )

    def run_once():
        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(_toy_data_by_participant())):
            return evaluate_cross_subject_hyperalignment("unused", [1, 2, 3, 4], config=config, label_shuffle_control=True, label_shuffle_seed=37)

    first = run_once()
    second = run_once()
    assert first["outer"] == second["outer"]
    assert first["predictions"] == second["predictions"]
    assert all(row["label_shuffle_control"] is True for row in first["outer"])
    assert {row["label_shuffle_seed"] for row in first["outer"]} == {37}
