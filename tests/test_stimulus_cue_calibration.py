import re
import unittest
from unittest.mock import patch

import numpy as np
from pymegdec.stimulus_cross_subject import CrossSubjectStimulusConfig
from pymegdec.stimulus_cue_calibration import (
    CueCalibrationConfig,
    evaluate_cross_subject_cue_calibrated_stimulus,
    load_participant_cue_calibration_features,
)
from tests.matlab_fixtures import cell_array


def _mat_data_from_patterns(labels, patterns, *, transform=None):
    labels = np.asarray(labels, dtype=int)
    canonical = {int(label): np.asarray(pattern, dtype=float) for label, pattern in patterns.items()}
    if transform is None:
        transform = np.eye(next(iter(canonical.values())).shape[0])
    transform = np.asarray(transform, dtype=float)
    time = np.asarray([-0.5, 0.0, 0.15, 0.2, 0.6], dtype=float)
    trials = []
    for repeat_index, label in enumerate(labels):
        pattern = transform @ canonical[int(label)]
        signal = np.zeros((pattern.shape[0], time.size), dtype=float)
        signal[:, (time >= 0.15) & (time <= 0.2)] = pattern[:, None]
        signal[:, (time >= -0.5) & (time <= 0.0)] = 0.01 * (repeat_index + 1)
        trials.append(signal)
    return {
        "trial": cell_array(trials),
        "time": cell_array([time for _ in trials]),
        "trialinfo": np.array([[labels]], dtype=object),
    }


def _loadmat_side_effect(main_by_participant, cue_by_participant):
    def loadmat(path):
        match = re.search(r"Part(\d+)(Cue)?Data\.mat$", str(path))
        if not match:
            raise AssertionError(f"Unexpected MAT path: {path}")
        participant = int(match.group(1))
        is_cue = bool(match.group(2))
        data = cue_by_participant[participant] if is_cue else main_by_participant[participant]
        return {"data": np.array([data], dtype=object)}

    return loadmat


class TestStimulusCueCalibration(unittest.TestCase):
    def test_load_participant_cue_calibration_features_uses_cue_data(self):
        patterns = {1: [1.0, 0.0], 2: [-1.0, 0.0]}
        main_by_participant = {1: _mat_data_from_patterns([1, 2, 1, 2], patterns)}
        cue_by_participant = {1: _mat_data_from_patterns([1, 2, 1, 2], patterns)}
        config = CrossSubjectStimulusConfig(
            window_center=0.175,
            window_size=0.1,
            normalization="none",
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cue_calibration.sio.loadmat", side_effect=_loadmat_side_effect(main_by_participant, cue_by_participant)) as loadmat:
            feature_set = load_participant_cue_calibration_features("unused", 1, config=config)

        self.assertEqual(feature_set.features.shape, (4, 2))
        self.assertEqual(feature_set.labels.tolist(), [1, 2, 1, 2])
        self.assertTrue(str(loadmat.call_args.args[0]).endswith("Part1CueData.mat"))

    def test_cue_calibrated_decoding_maps_held_out_subject_with_cue_data(self):
        labels = [1, 2, 3, 1, 2, 3]
        patterns = {
            1: [1.0, 0.0],
            2: [-0.5, 0.8660254038],
            3: [-0.5, -0.8660254038],
        }
        transforms = {
            1: np.eye(2),
            2: np.array([[0.0, -1.0], [1.0, 0.0]]),
            3: np.array([[0.0, 1.0], [1.0, 0.0]]),
        }
        main_by_participant = {participant: _mat_data_from_patterns(labels, patterns, transform=transform) for participant, transform in transforms.items()}
        cue_by_participant = {participant: _mat_data_from_patterns(labels, patterns, transform=transform) for participant, transform in transforms.items()}
        decode_config = CrossSubjectStimulusConfig(
            window_center=0.175,
            window_size=0.1,
            feature_mode="sensor_mean",
            normalization="none",
            classifier="multiclass-svm",
            classifier_param=0.5,
            components_pca=float("inf"),
            chance_classes=3,
            signflip_permutations=128,
        )
        calibration_config = CueCalibrationConfig(
            window_center=0.175,
            window_size=0.1,
            feature_mode="decode",
            normalization="decode",
        )

        loadmat = _loadmat_side_effect(main_by_participant, cue_by_participant)
        with (
            patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat),
            patch("pymegdec.stimulus_cue_calibration.sio.loadmat", side_effect=loadmat),
        ):
            artifacts = evaluate_cross_subject_cue_calibrated_stimulus(
                "unused",
                [1, 2, 3],
                decode_config=decode_config,
                calibration_config=calibration_config,
            )

        self.assertEqual(len(artifacts["outer"]), 3)
        self.assertEqual(len(artifacts["predictions"]), 18)
        self.assertEqual({row["alignment"] for row in artifacts["outer"]}, {"cue_class_procrustes"})
        self.assertEqual({row["calibration_data"] for row in artifacts["outer"]}, {"cue"})
        self.assertEqual({row["calibration_alignment"] for row in artifacts["outer"]}, {"cue_class_procrustes"})
        self.assertEqual({row["calibration_template_policy"] for row in artifacts["outer"]}, {"source_only"})
        self.assertEqual({row["target_calibration_label_shuffle_control"] for row in artifacts["outer"]}, {False})
        self.assertEqual({row["balanced_accuracy"] for row in artifacts["outer"]}, {1.0})
        self.assertEqual(artifacts["group_summary"][0]["alignment"], "cue_class_procrustes")
        self.assertEqual(artifacts["group_summary"][0]["calibration_feature_mode"], "sensor_mean")
        self.assertEqual(artifacts["group_summary"][0]["calibration_normalization"], "none")
        for row in artifacts["outer"]:
            aligned_participants = {int(token) for token in row["alignment_aligned_participants"].split(",")}
            self.assertEqual(aligned_participants, {1, 2, 3})
            self.assertEqual(int(row["target_calibration_participant"]), int(row["test_participant"]))

    def test_target_calibration_label_shuffle_control_marks_outputs(self):
        labels = [1, 2, 3, 1, 2, 3]
        patterns = {1: [1.0, 0.0], 2: [-0.5, 0.8660254038], 3: [-0.5, -0.8660254038]}
        main_by_participant = {participant: _mat_data_from_patterns(labels, patterns) for participant in (1, 2, 3)}
        cue_by_participant = {participant: _mat_data_from_patterns(labels, patterns) for participant in (1, 2, 3)}
        decode_config = CrossSubjectStimulusConfig(
            window_center=0.175,
            window_size=0.1,
            normalization="none",
            classifier="multiclass-svm",
            classifier_param=0.5,
            components_pca=float("inf"),
            chance_classes=3,
            signflip_permutations=128,
        )

        loadmat = _loadmat_side_effect(main_by_participant, cue_by_participant)
        with (
            patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=loadmat),
            patch("pymegdec.stimulus_cue_calibration.sio.loadmat", side_effect=loadmat),
        ):
            artifacts = evaluate_cross_subject_cue_calibrated_stimulus(
                "unused",
                [1, 2, 3],
                decode_config=decode_config,
                target_calibration_label_shuffle_control=True,
                target_calibration_label_shuffle_seed=7,
            )

        self.assertEqual({row["target_calibration_label_shuffle_control"] for row in artifacts["outer"]}, {True})
        self.assertEqual({row["target_calibration_label_shuffle_seed"] for row in artifacts["outer"]}, {7})
        self.assertEqual({row["target_calibration_label_shuffle_control"] for row in artifacts["predictions"]}, {True})
