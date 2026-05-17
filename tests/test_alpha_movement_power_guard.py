import numpy as np

from pymegdec import alpha_movement


def _cell(values):
    out = np.empty((1, len(values)), dtype=object)
    for index, value in enumerate(values):
        out[0, index] = value
    return out


def _minimal_data():
    return {
        "label": np.asarray(["M1", "M2"], dtype=object)[:, None],
        "trial": _cell([np.zeros((2, 2))]),
        "time": _cell([np.array([0.0, 0.02])]),
        "trialinfo": np.array([[1]]),
    }


def _geometry():
    return alpha_movement._MovementGeometry(
        channel_indices=np.array([0, 1]),
        channel_names=np.asarray(["M1", "M2"], dtype=object),
        positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        projected_positions=np.array([[0.0, 0.0], [10.0, 0.0]]),
    )


def _patch_geometry_and_power(monkeypatch, powers):
    monkeypatch.setattr(alpha_movement, "_selected_geometry", lambda *args, **kwargs: _geometry())
    monkeypatch.setattr(alpha_movement, "_alpha_power", lambda *args, **kwargs: np.asarray(powers, dtype=float))


def test_zero_alpha_power_marks_movement_row_unreliable(monkeypatch):
    _patch_geometry_and_power(monkeypatch, np.zeros((2, 2)))

    rows = alpha_movement.compute_alpha_movement_trajectory(
        _minimal_data(),
        0,
        channel_indices=np.array([0, 1]),
        config=alpha_movement.AlphaMovementConfig(time_window=(0.0, 0.02), trajectory_step_s=None),
    )

    assert len(rows) == 2
    assert rows[0]["total_alpha_power"] == 0.0
    assert rows[0]["peak_channel"] == -1
    assert rows[0]["peak_channel_name"] == ""
    assert np.isnan(rows[0]["centroid_x_mm"])
    assert np.isnan(rows[0]["projected_x_mm"])
    assert np.isnan(rows[0]["displacement_mm"])
    assert np.isnan(rows[0]["projected_direction_rad"])


def test_tiny_nonzero_alpha_power_is_not_epsilon_dominated(monkeypatch):
    _patch_geometry_and_power(monkeypatch, np.array([[0.0, 0.0], [1e-30, 1e-30]]))

    rows = alpha_movement.compute_alpha_movement_trajectory(
        _minimal_data(),
        0,
        channel_indices=np.array([0, 1]),
        config=alpha_movement.AlphaMovementConfig(time_window=(0.0, 0.02), trajectory_step_s=None),
    )

    assert len(rows) == 2
    np.testing.assert_allclose(rows[0]["projected_x_mm"], 10.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(rows[0]["centroid_x_mm"], 10.0, rtol=0.0, atol=1e-12)
    assert rows[0]["peak_channel"] == 1
    assert rows[0]["peak_channel_name"] == "M2"


def test_configured_alpha_power_floor_marks_low_power_unreliable(monkeypatch):
    _patch_geometry_and_power(monkeypatch, np.array([[0.0, 0.0], [0.5, 0.5]]))

    rows = alpha_movement.compute_alpha_movement_trajectory(
        _minimal_data(),
        0,
        channel_indices=np.array([0, 1]),
        config=alpha_movement.AlphaMovementConfig(
            time_window=(0.0, 0.02),
            trajectory_step_s=None,
            min_total_alpha_power=1.0,
        ),
    )

    assert len(rows) == 2
    assert rows[0]["total_alpha_power"] == 0.5
    assert np.isnan(rows[0]["projected_x_mm"])
    assert np.isnan(rows[0]["projected_direction_rad"])
