import neureptrace.bushmeg_supervised_lowrank_loso as nrt_lowrank
import neureptrace.decoding.mcca as nrt_mcca
import neureptrace.decoding.mcca_target as nrt_mcca_target
import neureptrace.decoding.windowed as nrt_windowed

from pymegdec import stimulus_mcca
from pymegdec.stimulus_full_epoch_lowrank_neureptrace import build_neureptrace_supervised_lowrank_config


def test_full_epoch_lowrank_config_targets_neureptrace_supervised_lowrank_schema():
    config = build_neureptrace_supervised_lowrank_config(
        data_folder="/data/bush",
        participants="1-2,4",
        time_windows=((0.0, 0.25), (-0.05, 0.25)),
        time_bin_size=0.025,
        temporal_feature_modes=("mean", "mean_d1"),
        classifier_params=(0.1, 1.0),
        components_values=(8, 16),
    )

    assert config["dataset"]["root"] == "/data/bush"
    assert config["participants"]["ids"] == "1-2,4"
    grid = config["supervised_lowrank_loso"]["candidate_grid"]
    assert grid["epoch_windows"][0]["start"] == 0.0
    assert grid["epoch_windows"][1]["stop"] == 0.25
    assert grid["temporal_bins"] == [10, 12]
    assert grid["pls_components"] == [8, 16]
    assert grid["include_deltas"] == [False, True]
    assert callable(nrt_lowrank.run_supervised_lowrank_loso)


def test_mcca_facade_binds_reusable_kernels_to_neureptrace():
    assert stimulus_mcca.fit_class_mcca is nrt_mcca.fit_class_mcca
    assert stimulus_mcca.class_alignment_matrix is nrt_mcca_target.class_alignment_matrix
    assert stimulus_mcca.fit_target_mcca_projection is nrt_mcca_target.fit_target_mcca_projection
    assert stimulus_mcca.fit_window_model is nrt_windowed.fit_window_model
    assert stimulus_mcca.predict_window_model is nrt_windowed.predict_window_model
    assert stimulus_mcca.transform_window_features is nrt_windowed.transform_window_features
