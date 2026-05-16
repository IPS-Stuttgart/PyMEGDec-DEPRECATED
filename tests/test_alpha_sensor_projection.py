import numpy as np

from pymegdec.alpha_metrics import (
    apply_sensor_projection,
    fit_sensor_projection,
    project_channel_positions,
    project_sensor_positions,
)


def _cell(values):
    out = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        out[0, index] = value
    return out


def _projection_data(positions, labels):
    return {
        "label": np.asarray(labels, dtype=object)[:, None],
        "trial": _cell([np.zeros((len(labels), 2))]),
        "grad": {"chanpos": np.asarray(positions, dtype=float)},
    }


def test_project_sensor_positions_anchors_axes_to_sensor_coordinates():
    positions = np.array(
        [
            [-20.0, 0.0, 7.0],
            [20.0, 0.0, 7.0],
            [0.0, 20.0, 7.0],
            [0.0, -20.0, 7.0],
        ]
    )

    projected = project_sensor_positions(positions)

    expected = positions[:, :2] - np.mean(positions[:, :2], axis=0)
    np.testing.assert_allclose(projected, expected, rtol=1e-12, atol=1e-12)


def test_project_sensor_positions_is_stable_after_sensor_reordering():
    positions = np.array(
        [
            [-30.0, -8.0, 3.0],
            [-10.0, 19.0, -1.0],
            [16.0, -6.0, 4.0],
            [24.0, 17.0, -2.0],
            [3.0, 5.0, 6.0],
        ]
    )
    permutation = np.array([2, 4, 0, 3, 1])

    projected = project_sensor_positions(positions)
    reordered_projected = project_sensor_positions(positions[permutation])

    inverse = np.argsort(permutation)
    np.testing.assert_allclose(reordered_projected[inverse], projected, rtol=1e-12, atol=1e-12)


def test_project_sensor_positions_uses_next_reference_axis_when_x_is_normal():
    positions = np.array(
        [
            [5.0, -20.0, 0.0],
            [5.0, 20.0, 0.0],
            [5.0, 0.0, 20.0],
            [5.0, 0.0, -20.0],
        ]
    )

    projected = project_sensor_positions(positions)

    expected = positions[:, [1, 2]] - np.mean(positions[:, [1, 2]], axis=0)
    np.testing.assert_allclose(projected, expected, rtol=1e-12, atol=1e-12)


def test_project_sensor_positions_keeps_positive_z_when_y_is_normal():
    positions = np.array(
        [
            [-20.0, 5.0, 0.0],
            [20.0, 5.0, 0.0],
            [0.0, 5.0, 20.0],
            [0.0, 5.0, -20.0],
        ]
    )

    projected = project_sensor_positions(positions)

    expected = positions[:, [0, 2]] - np.mean(positions[:, [0, 2]], axis=0)
    np.testing.assert_allclose(projected, expected, rtol=1e-12, atol=1e-12)


def test_fit_sensor_projection_skips_near_normal_reference_axis():
    positions = np.array(
        [
            [-0.02, -20.0, 0.0],
            [0.02, 20.0, 0.0],
            [0.0, 0.0, 20.0],
            [0.0, 0.0, -20.0],
        ]
    )

    projection = fit_sensor_projection(positions)

    assert projection.reference_projection_norms[0] < 0.05
    assert abs(float(np.dot(projection.axes[:, 0], np.array([0.0, 1.0, 0.0])))) > 0.99
    assert abs(float(np.dot(projection.axes[:, 1], np.array([0.0, 0.0, 1.0])))) > 0.99


def test_fit_and_apply_sensor_projection_reuses_same_basis_for_subsets():
    positions = np.array(
        [
            [-20.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [100.0, 20.0, 0.0],
        ]
    )

    projection = fit_sensor_projection(positions)
    projected_subset = apply_sensor_projection(positions[:3], projection)

    expected = positions[:3, :2] - np.mean(positions[:, :2], axis=0)
    np.testing.assert_allclose(projected_subset, expected, rtol=1e-12, atol=1e-12)


def test_project_channel_positions_uses_common_reference_sensor_set():
    positions = np.array(
        [
            [-20.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [100.0, 20.0, 0.0],
            [1000.0, 1000.0, 0.0],
        ]
    )
    data = _projection_data(
        positions,
        ["MLO11", "MRO11", "MZO01", "MLF11", "EEG001"],
    )
    selected = np.array([0, 1, 2])

    _, projected = project_channel_positions(data, selected, projection_reference_pattern=r"^M")
    _, selected_only_projected = project_channel_positions(data, selected, projection_reference_pattern=None)

    expected_common = positions[selected, :2] - np.mean(positions[:4, :2], axis=0)
    expected_selected_only = positions[selected, :2] - np.mean(positions[selected, :2], axis=0)
    np.testing.assert_allclose(projected, expected_common, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        selected_only_projected,
        expected_selected_only,
        rtol=1e-12,
        atol=1e-12,
    )
