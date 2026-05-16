"""Held-out subject handling for train-class Procrustes alignment."""

from __future__ import annotations

from dataclasses import replace

import numpy as np


def _alignment_model(cross_subject, alignment, *, common_classes, aligned_participants, transforms=(), target_transform=None):
    return {
        "metadata": cross_subject._alignment_metadata(
            alignment,
            common_classes=common_classes,
            aligned_participants=aligned_participants,
        ),
        "transforms": tuple(transforms),
        "target_transform": target_transform,
    }


def _group_average_channel_procrustes_transform(transforms):
    transforms = tuple(transforms)
    if not transforms:
        return None

    rotations = np.stack([np.asarray(transform["rotation"], dtype=float) for transform in transforms], axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    left, _singular_values, right_t = np.linalg.svd(mean_rotation, full_matrices=False)
    rotation = left @ right_t
    return {
        "source_center": np.mean(
            np.stack([np.asarray(transform["source_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "target_center": np.mean(
            np.stack([np.asarray(transform["target_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "rotation": rotation,
    }


def _fitted_alignment_model(fitted_model):
    alignment_metadata = fitted_model.get("alignment_metadata", {})
    if isinstance(alignment_metadata, dict) and "metadata" in alignment_metadata:
        return alignment_metadata
    return {
        "metadata": alignment_metadata,
        "transforms": tuple(),
        "target_transform": None,
    }


def _channel_feature_mean(cross_subject, features, feature_set):
    channel_features = cross_subject._features_as_trial_channel_matrix(features, feature_set)
    return np.mean(channel_features, axis=(0, 1))


def _target_centered_channel_procrustes_transform(cross_subject, target_transform, features, feature_set):
    return {
        "source_center": _channel_feature_mean(cross_subject, features, feature_set),
        "target_center": np.asarray(target_transform["target_center"], dtype=float),
        "rotation": np.asarray(target_transform["rotation"], dtype=float),
    }


def _test_alignment_metadata(test_transform, target_centering):
    return {"test_transform": test_transform, "target_centering": target_centering}


def _align_test_features_by_subject(cross_subject, test_features, test_set, config, alignment_model):
    if config.alignment == "none":
        return test_features, _test_alignment_metadata("none", "none")
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    target_transform = alignment_model.get("target_transform")
    if target_transform is None:
        return test_features, _test_alignment_metadata("none", "none")

    test_transform = _target_centered_channel_procrustes_transform(
        cross_subject,
        target_transform,
        test_features,
        test_set,
    )
    return (
        cross_subject._apply_channel_procrustes_transform(test_features, test_set, test_transform),
        _test_alignment_metadata("group_average_train_transform", "target_unsupervised"),
    )


def _prediction_group_columns_with_alignment(cross_subject):
    columns = tuple(cross_subject.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS)
    additions = ("alignment_test_transform", "alignment_target_centering")
    if all(column in columns for column in additions):
        return columns
    output = []
    for column in columns:
        output.append(column)
        if column == "alignment":
            output.extend(addition for addition in additions if addition not in output)
    return tuple(output)


def apply_procrustes_target_alignment():
    """Install target-side scoring alignment for ``train_class_procrustes``.

    The training-fold Procrustes option used to align only training subjects and
    then score raw held-out features.  This patch keeps the held-out labels
    untouched, but maps the held-out feature matrix into the same aligned
    channel space using the group-average training transform and the held-out
    subject's unlabeled feature mean for centering.
    """

    from pymegdec import stimulus_cross_subject as cross_subject

    if getattr(cross_subject, "_PROCRUSTES_TARGET_ALIGNMENT_PATCHED", False):
        return

    original_score_outer_fold_model = cross_subject._score_outer_fold_model

    def align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
        if config.alignment == "none":
            return features_by_subject, _alignment_model(
                cross_subject,
                config.alignment,
                common_classes=(),
                aligned_participants=(),
            )
        if config.alignment != "train_class_procrustes":
            raise ValueError(f"Unsupported alignment: {config.alignment}")

        common_classes = cross_subject._common_label_values(labels_by_subject)
        if len(common_classes) < 2:
            return features_by_subject, _alignment_model(
                cross_subject,
                config.alignment,
                common_classes=common_classes,
                aligned_participants=(),
            )

        class_patterns = [
            cross_subject._participant_class_channel_patterns(features, labels, feature_set, common_classes)
            for feature_set, features, labels in zip(feature_sets, features_by_subject, labels_by_subject, strict=True)
        ]
        transforms = cross_subject._fit_channel_procrustes_transforms(class_patterns)
        aligned_features = [
            cross_subject._apply_channel_procrustes_transform(features, feature_set, transform)
            for feature_set, features, transform in zip(feature_sets, features_by_subject, transforms, strict=True)
        ]
        return aligned_features, _alignment_model(
            cross_subject,
            config.alignment,
            common_classes=common_classes,
            aligned_participants=(feature_set.participant for feature_set in feature_sets),
            transforms=transforms,
            target_transform=_group_average_channel_procrustes_transform(transforms),
        )

    def score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
        alignment_model = _fitted_alignment_model(fitted_model)
        test_features = cross_subject._normalized_subject_features(test_set, config)
        test_features, test_alignment_metadata = _align_test_features_by_subject(
            cross_subject,
            test_features,
            test_set,
            config,
            alignment_model,
        )
        scoring_set = replace(test_set, features=test_features, normalization=config.normalization)
        scoring_model = dict(fitted_model)
        scoring_model["alignment_metadata"] = alignment_model["metadata"]
        outer_row, prediction_rows = original_score_outer_fold_model(
            scoring_model,
            scoring_set,
            config,
            include_predictions=include_predictions,
        )
        outer_row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
        outer_row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
        for row in prediction_rows:
            row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
            row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
        return outer_row, prediction_rows

    cross_subject._align_training_features_by_subject = align_training_features_by_subject
    cross_subject._score_outer_fold_model = score_outer_fold_model
    cross_subject.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = _prediction_group_columns_with_alignment(cross_subject)
    setattr(cross_subject, "_PROCRUSTES_TARGET_ALIGNMENT_PATCHED", True)
