import numpy as np

from pymegdec.alpha_metrics import project_sensor_positions


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
