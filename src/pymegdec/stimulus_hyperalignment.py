"""Public facade for Procrustes hyperalignment stimulus decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import sys

import numpy as np
from reptrace.decoding.hyperalignment import HyperalignmentModel, SubjectHyperalignmentProjection, class_alignment_matrices

from pymegdec import _stimulus_hyperalignment_legacy as _impl
from pymegdec._reptrace_score_overrides import install_hyperalignment

HYPERALIGNMENT_INITIALIZATION_MODES = ("pca", "mean")

_ORIGINAL_EVALUATE_HYPERALIGNMENT_OUTER_FOLD = _impl._evaluate_hyperalignment_outer_fold
_ORIGINAL_NORMALIZED_HYPERALIGNMENT_CONFIG = _impl._normalized_hyperalignment_config


def _normalize_hyperalignment_initialization(initialization):
    normalized = str(initialization).strip().lower().replace("-", "_")
    if normalized not in HYPERALIGNMENT_INITIALIZATION_MODES:
        raise ValueError(
            f"Unsupported hyperalignment initialization: {initialization}. "
            f"Supported modes: {', '.join(HYPERALIGNMENT_INITIALIZATION_MODES)}."
        )
    return normalized


def _normalized_hyperalignment_config(config):
    normalized = _ORIGINAL_NORMALIZED_HYPERALIGNMENT_CONFIG(config)
    initialization = _normalize_hyperalignment_initialization(normalized.hyperalignment_initialization)
    if initialization == normalized.hyperalignment_initialization:
        return normalized
    return replace(normalized, hyperalignment_initialization=initialization)


def _evaluate_hyperalignment_outer_fold(*args, **kwargs):
    config = args[4] if len(args) >= 5 else kwargs.get("config")
    initialization = _normalize_hyperalignment_initialization(config.hyperalignment_initialization)
    if initialization == "pca":
        return _ORIGINAL_EVALUATE_HYPERALIGNMENT_OUTER_FOLD(*args, **kwargs)

    original_fit_class_hyperalignment = _impl.fit_class_hyperalignment
    _impl.fit_class_hyperalignment = _fit_mean_initialized_class_hyperalignment
    try:
        return _ORIGINAL_EVALUATE_HYPERALIGNMENT_OUTER_FOLD(*args, **kwargs)
    finally:
        _impl.fit_class_hyperalignment = original_fit_class_hyperalignment


def _fit_mean_initialized_class_hyperalignment(
    features_by_subject,
    labels_by_subject,
    *,
    sample_mode="class_mean",
    n_repetitions_per_class=None,
    n_components=64,
    n_iterations=10,
    template_tolerance=1e-8,
):
    alignment = class_alignment_matrices(
        features_by_subject,
        labels_by_subject,
        sample_mode=sample_mode,
        n_repetitions_per_class=n_repetitions_per_class,
    )
    model = _fit_mean_initialized_hyperalignment(
        alignment.aligned_by_subject,
        n_components=n_components,
        n_iterations=n_iterations,
        template_tolerance=template_tolerance,
    )
    return model, alignment


def _fit_mean_initialized_hyperalignment(
    aligned_by_subject: Mapping[object, object],
    *,
    n_components,
    n_iterations,
    template_tolerance: float = 1e-8,
) -> HyperalignmentModel:
    """Fit Procrustes hyperalignment from a grand-mean initialized template."""

    if len(aligned_by_subject) < 2:
        raise ValueError("Hyperalignment requires at least two subjects.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")

    subject_ids = tuple(aligned_by_subject.keys())
    matrices = {subject_id: _hyperalignment_feature_matrix(matrix, name=f"aligned_by_subject[{subject_id!r}]") for subject_id, matrix in aligned_by_subject.items()}
    n_rows = _hyperalignment_common_row_count(matrices)
    if n_rows < 2:
        raise ValueError("Hyperalignment requires at least two aligned rows per subject.")

    feature_dims = {matrix.shape[1] for matrix in matrices.values()}
    if len(feature_dims) != 1:
        raise ValueError("Mean hyperalignment initialization requires all subjects to have the same feature dimension.")

    requested = _hyperalignment_requested_component_count(n_components)
    actual = min(requested, n_rows - 1, next(iter(feature_dims)))
    if actual < 1:
        raise ValueError("No hyperalignment components are available.")

    means = {subject_id: np.mean(matrix, axis=0) for subject_id, matrix in matrices.items()}
    centered = {subject_id: matrices[subject_id] - means[subject_id] for subject_id in subject_ids}
    mean_centered = np.mean(np.stack([centered[subject_id] for subject_id in subject_ids], axis=0), axis=0)
    mean_projection = _hyperalignment_initial_projection(mean_centered, actual)
    template = _hyperalignment_normalize_template(mean_centered @ mean_projection)
    projections = {subject_id: _hyperalignment_procrustes_projection(centered[subject_id], template) for subject_id in subject_ids}

    for _ in range(int(n_iterations)):
        new_projections = {subject_id: _hyperalignment_procrustes_projection(centered[subject_id], template) for subject_id in subject_ids}
        new_template = _hyperalignment_normalize_template(
            np.mean(np.stack([centered[subject_id] @ new_projections[subject_id] for subject_id in subject_ids], axis=0), axis=0)
        )
        delta = float(np.linalg.norm(new_template - template) / max(np.linalg.norm(template), 1e-12))
        projections = new_projections
        template = new_template
        if delta < template_tolerance:
            break

    projection_objects = {
        subject_id: SubjectHyperalignmentProjection(
            subject_id=subject_id,
            feature_mean=means[subject_id],
            projection=projections[subject_id],
            n_alignment_rows=n_rows,
        )
        for subject_id in subject_ids
    }
    group_feature_mean, group_projection = _hyperalignment_average_projection(projection_objects)
    return HyperalignmentModel(
        subject_ids=subject_ids,
        n_components=actual,
        n_iterations=int(n_iterations),
        projections=projection_objects,
        template=template,
        group_feature_mean=group_feature_mean,
        group_projection=group_projection,
    )


def _hyperalignment_feature_matrix(features, *, name):
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _hyperalignment_common_row_count(matrices):
    row_counts = {subject_id: matrix.shape[0] for subject_id, matrix in matrices.items()}
    unique_counts = set(row_counts.values())
    if len(unique_counts) != 1:
        raise ValueError(f"All subject alignment matrices must have the same row count, got {row_counts}.")
    return int(next(iter(unique_counts)))


def _hyperalignment_requested_component_count(n_components):
    if n_components == float("inf"):
        return np.iinfo(np.int32).max
    requested = int(n_components)
    if requested < 1:
        raise ValueError("n_components must be positive or infinity.")
    return requested


def _hyperalignment_initial_projection(centered, n_components):
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    projection = vt[:n_components].T
    if projection.shape[1] < n_components:
        projection = np.pad(projection, ((0, 0), (0, n_components - projection.shape[1])))
    return projection


def _hyperalignment_procrustes_projection(centered, template):
    u, _s, vt = np.linalg.svd(centered.T @ template, full_matrices=False)
    return u @ vt


def _hyperalignment_normalize_template(template):
    template = template - np.mean(template, axis=0, keepdims=True)
    scale = np.std(template, axis=0, ddof=1)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return template / scale[None, :]


def _hyperalignment_average_projection(projections: Mapping[object, SubjectHyperalignmentProjection]):
    feature_dims = {projection.projection.shape[0] for projection in projections.values()}
    if len(feature_dims) != 1:
        return None, None
    mean = np.mean(np.stack([projection.feature_mean for projection in projections.values()], axis=0), axis=0)
    matrix = np.mean(np.stack([projection.projection for projection in projections.values()], axis=0), axis=0)
    return mean, matrix


_impl.HYPERALIGNMENT_INITIALIZATION_MODES = HYPERALIGNMENT_INITIALIZATION_MODES
_impl._normalize_hyperalignment_initialization = _normalize_hyperalignment_initialization
_impl._normalized_hyperalignment_config = _normalized_hyperalignment_config
_impl._fit_mean_initialized_class_hyperalignment = _fit_mean_initialized_class_hyperalignment
_impl._fit_mean_initialized_hyperalignment = _fit_mean_initialized_hyperalignment
_impl._evaluate_hyperalignment_outer_fold = _evaluate_hyperalignment_outer_fold
install_hyperalignment(_impl)

sys.modules[__name__] = _impl
globals().update(_impl.__dict__)
