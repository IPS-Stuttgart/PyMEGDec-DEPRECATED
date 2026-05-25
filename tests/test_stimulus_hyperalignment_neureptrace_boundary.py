import neureptrace.decoding.hyperalignment as nrt_hyperalignment
import neureptrace.decoding.windowed as nrt_windowed

from pymegdec import stimulus_hyperalignment as hyperalignment


def test_hyperalignment_facade_binds_reusable_kernels_to_neureptrace():
    assert hyperalignment.class_alignment_matrices is nrt_hyperalignment.class_alignment_matrices
    assert hyperalignment.fit_class_hyperalignment is nrt_hyperalignment.fit_class_hyperalignment
    assert hyperalignment.fit_projection_to_hyperalignment is nrt_hyperalignment.fit_projection_to_hyperalignment
    assert hyperalignment.fit_window_model is nrt_windowed.fit_window_model
    assert hyperalignment.predict_window_model is nrt_windowed.predict_window_model
    assert hyperalignment.transform_window_features is nrt_windowed.transform_window_features


def test_hyperalignment_config_normalization_uses_neureptrace_initialization_modes():
    config = hyperalignment.CrossSubjectHyperalignmentConfig(hyperalignment_initialization="mean")
    normalized = hyperalignment._normalized_hyperalignment_config(config)  # pylint: disable=protected-access
    assert normalized.hyperalignment_initialization == "mean"
    assert "pca" in hyperalignment.HYPERALIGNMENT_INITIALIZATION_MODES
    assert "mean" in hyperalignment.HYPERALIGNMENT_INITIALIZATION_MODES
