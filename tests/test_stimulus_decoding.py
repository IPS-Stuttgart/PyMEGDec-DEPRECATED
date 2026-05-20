import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from pymegdec.stimulus_decoding import (
    StimulusDecodingConfig,
    _annotate_stimulus_onset_scan_with_reptrace,
    _stimulus_onset_event_rows_from_reptrace,
    evaluate_participant_stimulus_decoding_diagnostics,
    evaluate_participant_stimulus_onset_scan,
    evaluate_participant_stimulus_temporal_generalization,
    evaluate_participant_time_resolved_stimulus_transfer,
    summarize_stimulus_decoding,
    summarize_stimulus_decoding_peaks,
    summarize_stimulus_onset_events,
    summarize_stimulus_onset_scan,
    summarize_stimulus_prediction_diagnostics,
    summarize_stimulus_temporal_generalization,
    window_centers_from_range,
)
from tests.matlab_fixtures import cell_array


def _mat_data(labels, trial_values, time):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.asarray(labels, dtype=int)
    return {
        "trial": cell_array([np.asarray([[0.0, value]], dtype=float) for value in trial_values]),
        "time": cell_array([np.asarray([time], dtype=float) for _ in trial_values]),
        "trialinfo": trialinfo,
    }


def _mat_data_matrix(labels, trial_values, time):
    trialinfo = np.empty((1, 1), dtype=object)
    trialinfo[0, 0] = np.asarray(labels, dtype=int)
    return {
        "trial": cell_array([np.asarray([values], dtype=float) for values in trial_values]),
        "time": cell_array([np.asarray([time], dtype=float) for _ in trial_values]),
        "trialinfo": trialinfo,
    }


def _onset_scan_row(time, score, *, predicted_label=1, trial=0):
    scan_window = (time - 0.0125, time + 0.0125)
    correct = predicted_label == 1
    return {
        "participant": 1,
        "variant": "without_null",
        "transfer_direction": "main-to-cue",
        "train_window_center_s": 0.175,
        "train_window_start_s": 0.125,
        "train_window_stop_s": 0.225,
        "scan_window_center_s": time,
        "scan_window_start_s": scan_window[0],
        "scan_window_stop_s": scan_window[1],
        "trial": trial,
        "validation_trial_index": trial,
        "validation_trial_number": trial + 1,
        "true_label": 1,
        "predicted_label": predicted_label,
        "true_stimulus": 1,
        "predicted_stimulus": predicted_label,
        "true_stimulus_id": 1,
        "predicted_stimulus_id": predicted_label,
        "correct": correct,
        "stimulus_score": score,
        "score_threshold": np.nan,
        "above_threshold": False,
        "threshold_quantile": 0.0,
        "threshold_window_start_s": -0.10,
        "threshold_window_stop_s": -0.05,
        "chance_accuracy": 0.5,
        "chance_percent": 50.0,
        "classifier": "multiclass-svm",
        "classifier_param": 0.5,
        "components_pca": 100,
        "actual_components_pca": 1,
        "pca_explained_variance_percent": 100.0,
        "frequency_low_hz": 0.0,
        "frequency_high_hz": float("inf"),
    }


class TestStimulusDecoding(unittest.TestCase):
    def test_window_centers_from_range_includes_stop(self):
        self.assertEqual(
            window_centers_from_range((-0.1, 0.1), 0.1),
            (-0.1, 0.0, 0.1),
        )

    def test_evaluate_participant_time_resolved_stimulus_transfer(self):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ) as loadmat:
            rows = evaluate_participant_time_resolved_stimulus_transfer("unused", 1, config=config)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["variant"], "without_null")
        self.assertEqual(rows[0]["transfer_direction"], "main-to-cue")
        self.assertEqual(rows[0]["accuracy"], 1.0)
        self.assertEqual(rows[0]["chance_accuracy"], 0.5)
        self.assertTrue(str(loadmat.call_args_list[0].args[0]).endswith("Part1Data.mat"))
        self.assertTrue(str(loadmat.call_args_list[1].args[0]).endswith("Part1CueData.mat"))

    def test_evaluate_participant_stimulus_decoding_diagnostics(self):
        labels = [1, 2, 1, 2]
        train_data = _mat_data_matrix(labels, [[0.0, -2.0, -2.0], [0.0, 2.0, 2.0], [0.0, -1.0, -1.0], [0.0, 1.0, 1.0]], [-0.1, 0.0, 0.1])
        validation_data = _mat_data_matrix(labels, [[0.0, -1.5, -1.5], [0.0, 1.5, 1.5], [0.0, -0.5, -0.5], [0.0, 0.5, 0.5]], [-0.1, 0.0, 0.1])
        config = StimulusDecodingConfig(
            window_centers=(0.0, 0.1),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            rows, prediction_rows = evaluate_participant_stimulus_decoding_diagnostics(
                "unused",
                1,
                config=config,
                diagnostic_window_centers=(0.0,),
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(prediction_rows), 4)
        self.assertEqual({row["window_center_s"] for row in prediction_rows}, {0.0})
        self.assertEqual([row["true_stimulus"] for row in prediction_rows], labels)
        self.assertEqual([row["true_stimulus_id"] for row in prediction_rows], labels)
        self.assertEqual({row["transfer_direction"] for row in prediction_rows}, {"main-to-cue"})
        self.assertEqual([row["validation_trial_index"] for row in prediction_rows], [0, 1, 2, 3])
        self.assertEqual([row["validation_trial_number"] for row in prediction_rows], [1, 2, 3, 4])
        self.assertEqual({row["actual_components_pca"] for row in prediction_rows}, {1})
        self.assertTrue(all(row["correct"] for row in prediction_rows))

    def test_cue_to_main_transfer_swaps_train_and_validation_files(self):
        labels = [1, 2, 1, 2]
        cue_train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        main_validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
            transfer_direction="cue-to-main",
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([cue_train_data], dtype=object)},
                {"data": np.array([main_validation_data], dtype=object)},
            ],
        ) as loadmat:
            rows, prediction_rows = evaluate_participant_stimulus_decoding_diagnostics(
                "unused",
                1,
                config=config,
                diagnostic_window_centers=(0.0,),
            )

        self.assertTrue(str(loadmat.call_args_list[0].args[0]).endswith("Part1CueData.mat"))
        self.assertTrue(str(loadmat.call_args_list[1].args[0]).endswith("Part1Data.mat"))
        self.assertEqual(rows[0]["transfer_direction"], "cue-to-main")
        self.assertEqual({row["transfer_direction"] for row in prediction_rows}, {"cue-to-main"})
        self.assertEqual([row["true_stimulus_id"] for row in prediction_rows], labels)
        self.assertTrue(all(row["correct"] for row in prediction_rows))

    # jscpd:ignore-start
    def test_evaluate_participant_stimulus_temporal_generalization(self):
        labels = [1, 2, 1, 2]
        train_data = _mat_data_matrix(labels, [[-2.0, -2.0], [2.0, 2.0], [-1.0, -1.0], [1.0, 1.0]], [0.0, 0.1])
        validation_data = _mat_data_matrix(labels, [[-1.5, -1.5], [1.5, 1.5], [-0.5, -0.5], [0.5, 0.5]], [0.0, 0.1])
        config = StimulusDecodingConfig(
            window_centers=(0.0, 0.1),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            rows = evaluate_participant_stimulus_temporal_generalization("unused", 1, config=config)

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["train_window_center_s"] for row in rows}, {0.0, 0.1})
        self.assertEqual({row["test_window_center_s"] for row in rows}, {0.0, 0.1})
        self.assertEqual({row["accuracy"] for row in rows}, {1.0})
        self.assertEqual(sum(row["is_diagonal"] for row in rows), 2)
        self.assertEqual({row["actual_components_pca"] for row in rows}, {1})

    def test_summarize_stimulus_temporal_generalization(self):
        rows = [
            {
                "variant": "without_null",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.0,
                "accuracy": 0.25,
                "chance_accuracy": 0.0625,
            },
            {
                "variant": "without_null",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.0,
                "accuracy": 0.5,
                "chance_accuracy": 0.0625,
            },
            {
                "variant": "without_null",
                "train_window_center_s": 0.0,
                "test_window_center_s": 0.1,
                "accuracy": 0.1,
                "chance_accuracy": 0.0625,
            },
        ]

        summary = summarize_stimulus_temporal_generalization(rows)

        self.assertEqual(len(summary), 2)
        diagonal = [row for row in summary if row["is_diagonal"]][0]
        self.assertEqual(diagonal["n_participants"], 2)
        self.assertAlmostEqual(diagonal["accuracy_mean"], 0.375)
        self.assertEqual(diagonal["above_chance_count"], 2)

    # jscpd:ignore-end
    # jscpd:ignore-start
    def test_evaluate_participant_stimulus_onset_scan(self):
        labels = [1, 2, 1, 2]
        train_data = _mat_data_matrix(labels, [[-2.0, -2.0], [2.0, 2.0], [-1.0, -1.0], [1.0, 1.0]], [-0.1, 0.0])
        validation_data = _mat_data_matrix(labels, [[-1.5, -1.5], [1.5, 1.5], [-0.5, -0.5], [0.5, 0.5]], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(-0.1, 0.0),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
        )

        with patch(
            "pymegdec.stimulus_decoding.sio.loadmat",
            side_effect=[
                {"data": np.array([train_data], dtype=object)},
                {"data": np.array([validation_data], dtype=object)},
            ],
        ):
            scan_rows, event_rows = evaluate_participant_stimulus_onset_scan(
                "unused",
                1,
                config=config,
                train_window_center=0.0,
                threshold_window=(-0.1, -0.1),
                threshold_quantile=0.0,
                detection_start_s=0.0,
            )

        self.assertEqual(len(scan_rows), 8)
        self.assertEqual(len(event_rows), 4)
        self.assertEqual({row["scan_window_center_s"] for row in scan_rows}, {-0.1, 0.0})
        self.assertTrue(all(np.isfinite(row["stimulus_score"]) for row in scan_rows))
        self.assertTrue(all(np.isfinite(row["score_threshold"]) for row in scan_rows))
        self.assertTrue(all(row["detected"] for row in event_rows))
        self.assertTrue(all(row["detection_window_center_s"] == 0.0 for row in event_rows))
        self.assertTrue(all(row["correct_detected_stimulus"] for row in event_rows))

    def test_onset_scan_sustained_run_rejects_one_bin_spike(self):
        scan_rows = [
            _onset_scan_row(-0.10, 0.10),
            _onset_scan_row(-0.05, 0.10),
            _onset_scan_row(0.00, 2.00),
            _onset_scan_row(0.05, 0.00),
        ]

        thresholded = _annotate_stimulus_onset_scan_with_reptrace(
            scan_rows,
            threshold_window=(-0.10, -0.05),
            threshold_quantile=0.0,
            threshold_method="max_run",
            min_consecutive=2,
        )
        event_rows = _stimulus_onset_event_rows_from_reptrace(
            thresholded,
            threshold_window=(-0.10, -0.05),
            threshold_quantile=0.0,
            threshold_method="max_run",
            min_consecutive=2,
            detection_start_s=0.0,
        )

        self.assertEqual(len(event_rows), 1)
        self.assertFalse(event_rows[0]["detected"])
        self.assertEqual(event_rows[0]["min_consecutive"], 2)
        self.assertEqual(event_rows[0]["threshold_method"], "max_run")

    def test_onset_scan_stable_prediction_requirement_breaks_class_flips(self):
        scan_rows = [
            _onset_scan_row(-0.10, 0.10, predicted_label=1),
            _onset_scan_row(-0.05, 0.10, predicted_label=1),
            _onset_scan_row(0.00, 2.00, predicted_label=1),
            _onset_scan_row(0.05, 2.00, predicted_label=2),
        ]

        thresholded = _annotate_stimulus_onset_scan_with_reptrace(
            scan_rows,
            threshold_window=(-0.10, -0.05),
            threshold_quantile=0.0,
            threshold_method="max_run",
            min_consecutive=2,
            require_stable_prediction=True,
        )
        event_rows = _stimulus_onset_event_rows_from_reptrace(
            thresholded,
            threshold_window=(-0.10, -0.05),
            threshold_quantile=0.0,
            threshold_method="max_run",
            min_consecutive=2,
            require_stable_prediction=True,
            detection_start_s=0.0,
        )

        self.assertEqual(len(event_rows), 1)
        self.assertFalse(event_rows[0]["detected"])
        self.assertTrue(event_rows[0]["require_stable_prediction"])

    def test_summarize_stimulus_onset_scan_and_events(self):
        scan_rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.175,
                "scan_window_center_s": -0.1,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "correct": False,
                "stimulus_score": 0.5,
                "above_threshold": False,
                "score_threshold": 1.0,
                "threshold_quantile": 0.95,
                "threshold_window_start_s": -0.35,
                "threshold_window_stop_s": -0.05,
                "chance_accuracy": 0.0625,
                "chance_percent": 6.25,
            },
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.175,
                "scan_window_center_s": -0.1,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "correct": True,
                "stimulus_score": 1.5,
                "above_threshold": True,
                "score_threshold": 1.0,
                "threshold_quantile": 0.95,
                "threshold_window_start_s": -0.35,
                "threshold_window_stop_s": -0.05,
                "chance_accuracy": 0.0625,
                "chance_percent": 6.25,
            },
        ]
        event_rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "transfer_direction": "main-to-cue",
                "train_window_center_s": 0.175,
                "classifier": "multiclass-svm",
                "components_pca": 100,
                "frequency_low_hz": 0.0,
                "frequency_high_hz": float("inf"),
                "detected": True,
                "detected_before_stimulus": False,
                "correct_detected_stimulus": True,
                "detection_latency_s": 0.175,
                "score_threshold": 1.0,
                "threshold_quantile": 0.95,
                "threshold_window_start_s": -0.35,
                "threshold_window_stop_s": -0.05,
            }
        ]

        scan_summary = summarize_stimulus_onset_scan(scan_rows)
        event_summary = summarize_stimulus_onset_events(event_rows)

        self.assertEqual(len(scan_summary), 1)
        self.assertAlmostEqual(scan_summary[0]["accuracy"], 0.5)
        self.assertAlmostEqual(scan_summary[0]["above_threshold_rate"], 0.5)
        self.assertEqual(len(event_summary), 1)
        self.assertAlmostEqual(event_summary[0]["detected_rate"], 1.0)
        self.assertAlmostEqual(event_summary[0]["post_detection_latency_mean_s"], 0.175)

    # jscpd:ignore-end
    def test_evaluate_participant_time_resolved_stimulus_transfer_with_permutations(
        self,
    ):
        labels = [1, 2, 1, 2]
        train_data = _mat_data(labels, [-2.0, 2.0, -1.0, 1.0], [-0.1, 0.0])
        validation_data = _mat_data(labels, [-1.5, 1.5, -0.5, 0.5], [-0.1, 0.0])
        config = StimulusDecodingConfig(
            window_centers=(0.0,),
            window_size=0.0,
            components_pca=float("inf"),
            chance_classes=2,
            permutations=3,
            permutation_seed=0,
        )

        with (
            patch(
                "pymegdec.stimulus_decoding.sio.loadmat",
                side_effect=[
                    {"data": np.array([train_data], dtype=object)},
                    {"data": np.array([validation_data], dtype=object)},
                ],
            ),
            patch(
                "pymegdec._stimulus_decoding_core.evaluate_feature_transfer",
                return_value=SimpleNamespace(
                    model_bundle=SimpleNamespace(
                        actual_components_pca=2,
                        explained_variance_percent=100.0,
                    ),
                    predictions=np.array([1, 2, 1, 2]),
                    accuracy=0.75,
                    permutation_accuracy=np.array([0.0, 0.25, 0.5]),
                    permutation_p_value=0.25,
                ),
            ),
        ):
            rows = evaluate_participant_time_resolved_stimulus_transfer("unused", 1, config=config)

        self.assertEqual(rows[0]["n_permutations"], 3)
        self.assertAlmostEqual(rows[0]["permutation_p_value"], 0.25)
        self.assertAlmostEqual(rows[0]["permutation_accuracy_mean"], 0.25)

    def test_summarize_stimulus_decoding(self):
        rows = [
            {
                "variant": "without_null",
                "window_center_s": 0.0,
                "accuracy": 0.25,
                "chance_accuracy": 0.0625,
                "permutation_p_value": 0.04,
            },
            {
                "variant": "without_null",
                "window_center_s": 0.0,
                "accuracy": 0.5,
                "chance_accuracy": 0.0625,
                "permutation_p_value": 0.006,
            },
            {
                "variant": "without_null",
                "window_center_s": 0.1,
                "accuracy": 0.5,
                "chance_accuracy": 0.0625,
                "permutation_p_value": np.nan,
            },
        ]

        summary = summarize_stimulus_decoding(rows)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["n_participants"], 2)
        self.assertAlmostEqual(summary[0]["accuracy_mean"], 0.375)
        self.assertEqual(summary[0]["above_chance_count"], 2)
        self.assertEqual(summary[0]["n_with_permutation"], 2)
        self.assertEqual(summary[0]["n_significant_p_0.05"], 2)
        self.assertEqual(summary[0]["n_significant_p_0.01"], 1)

    def test_summarize_stimulus_decoding_peaks(self):
        rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.1,
                "window_start_s": 0.05,
                "window_stop_s": 0.15,
                "accuracy": 0.2,
                "percent": 20.0,
                "chance_accuracy": 0.0625,
                "chance_percent": 6.25,
            },
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.2,
                "window_start_s": 0.15,
                "window_stop_s": 0.25,
                "accuracy": 0.3,
                "percent": 30.0,
                "chance_accuracy": 0.0625,
                "chance_percent": 6.25,
            },
        ]

        peaks = summarize_stimulus_decoding_peaks(rows)

        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0]["peak_window_center_s"], 0.2)
        self.assertEqual(peaks[0]["peak_accuracy"], 0.3)

    def test_summarize_stimulus_prediction_diagnostics(self):
        prediction_rows = [
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.2,
                "true_stimulus": 1,
                "predicted_stimulus": 1,
                "correct": True,
            },
            {
                "participant": 1,
                "variant": "without_null",
                "window_center_s": 0.2,
                "true_stimulus": 1,
                "predicted_stimulus": 2,
                "correct": False,
            },
        ]

        confusion, per_stimulus = summarize_stimulus_prediction_diagnostics(prediction_rows)

        self.assertEqual(len(confusion), 2)
        self.assertEqual(per_stimulus[0]["true_stimulus"], 1)
        self.assertEqual(per_stimulus[0]["n_trials"], 2)
        self.assertEqual(per_stimulus[0]["n_correct"], 1)
        self.assertEqual(per_stimulus[0]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
