"""Compatibility helper for alpha movement's shared sensor projection frame."""

from __future__ import annotations

import numpy as np

from pymegdec.alpha_metrics import (
    DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
    DEFAULT_PROJECTION_REFERENCE_PATTERN,
    DEFAULT_SENSOR_POSITION_UNIT,
    get_channel_names,
    project_channel_positions,
)


def _selected_geometry_common_projection(
    data,
    trial_signal,
    channel_indices,
    sensor_position_unit=DEFAULT_SENSOR_POSITION_UNIT,
    projection_reference_pattern=DEFAULT_PROJECTION_REFERENCE_PATTERN,
    min_reference_axis_projection=DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
):
    import pymegdec.alpha_movement as alpha_movement

    positions, projected_positions = project_channel_positions(
        data,
        channel_indices,
        sensor_position_unit=sensor_position_unit,
        projection_reference_pattern=projection_reference_pattern,
        min_reference_axis_projection=min_reference_axis_projection,
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
    """Return the direct common-projection geometry function.

    Kept for compatibility with older callers that explicitly imported this
    helper. The public ``alpha_movement`` module now implements the same
    common-frame projection directly and no longer needs package-import
    monkey-patching.
    """

    import pymegdec.alpha_movement as alpha_movement

    return alpha_movement._selected_geometry
