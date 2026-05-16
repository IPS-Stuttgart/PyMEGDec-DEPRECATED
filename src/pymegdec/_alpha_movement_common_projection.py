"""Patch alpha movement to use the shared sensor projection frame."""

from __future__ import annotations

import numpy as np

from pymegdec.alpha_metrics import (
    DEFAULT_SENSOR_POSITION_UNIT,
    get_channel_names,
    project_channel_positions,
)


def _selected_geometry_common_projection(
    data,
    trial_signal,
    channel_indices,
    sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT,
):
    import pymegdec.alpha_movement as alpha_movement

    positions, projected_positions = project_channel_positions(
        data,
        channel_indices,
        sensor_position_unit=sensor_position_unit,
    )
    channel_names = np.asarray(get_channel_names(data, trial_signal.shape[0]), dtype=object)[
        channel_indices
    ]
    return alpha_movement._MovementGeometry(
        channel_indices=channel_indices,
        channel_names=channel_names,
        positions=positions,
        projected_positions=projected_positions,
    )


def apply_alpha_movement_common_projection():
    """Make alpha movement use the same common-frame projection as alpha metrics."""

    import pymegdec.alpha_movement as alpha_movement

    alpha_movement._selected_geometry = _selected_geometry_common_projection
    return _selected_geometry_common_projection
