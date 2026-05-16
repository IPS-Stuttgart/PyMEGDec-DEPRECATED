"""Cue-calibrated cross-subject stimulus decoding."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.io as sio
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec import stimulus_cross_subject as cross_subject
from pymegdec.stimulus_cross_subject import (
    DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
    DEFAULT_CROSS_SUBJECT_WINDOW_CENTER,
    DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
    CrossSubjectStimulusConfig,
    ParticipantFeatureSet,
)
from reptrace.decoding.windowed import fit_window_model as fit_reptrace_window_model

DEFAULT_CUE_CALIBRATION_ALIGNMENT = "cue_class_procrustes"
DEFAULT_CUE_CALIBRATION_DATA = "cue"
DEFAULT_CUE_CALIBRATION_TEMPLATE_POLICY = "source_only"
CUE_CALIBRATION_ALIGNMENTS = (DEFAULT_CUE_CALIBRATION_ALIGNMENT,)
CUE_CALIBRATION_DATASETS = (DEFAULT_CUE_CALIBRATION_DATA,)
CUE_CALIBRATION_TEMPLATE_POLICIES = (DEFAULT_CUE_CALIBRATION_TEMPLATE_POLICY,)
DECODE_REFERENCE_TOKEN = "decode"


@dataclass(frozen=True)
class CueCalibrationConfig:
    """Parameters for an auxiliary cue/localizer alignment run.

    The calibration transform is fitted from ``Part*CueData.mat`` and then applied to
    the main-task ``Part*Data.mat`` features.  ``feature_mode`` and
    ``normalization`` may be set to ``"decode"`` to reuse the main decoding
    configuration.
    """

    calibration_data: str = DEFAULT_CUE_CALIBRATION_DATA
    alignment: str = DEFAULT_CUE_CALIBRATION_ALIGNMENT
    template_policy: str = DEFAULT_CUE_CALIBRATION_TEMPLATE_POLICY
    window_center: float = DEFAULT_CROSS_SUBJECT_WINDOW_CENTER
    window_size: float = DEFAULT_CROSS_SUBJECT_WINDOW_SIZE
    baseline_window: tuple[float, float] = DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW
    feature_mode: str = DECODE_REFERENCE_TOKEN
    normalization: str = DECODE_REFERENCE_TOKEN
    max_trials_per_class_per_participant: int | None = None


def evaluate_cross_subject_cue_calibrated_stimulus(  # pylint: disable=too-many-arguments,too-many-locals
    data_folder,
    participants,
    *,
    decode_config=None,
    calibration_config=None,
    outer_participants=None,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
    target_calibration_label_shuffle_control=False,
    target_calibration_label_shuffle_seed=0,
):
    """Run LOSO main-task decoding after cue/localizer Procrustes calibration.

    Source participants' cue data define a source-only template.  Source and
    held-out participants are mapped into that template using cue data only; the
    classifier is trained on source participants' aligned main-task trials and
    scored on the held-out participant's aligned main-task trials.
    """

    decode_config = _normalized_decode_config(decode_config or CrossSubjectStimulusConfig())
    calibration_config = _normalized_cue_calibration_config(calibration_config or CueCalibrationConfig(), decode_config)
    data_folder = resolve_data_folder(data_folder)
    participants = tuple(int(participant) for participant in participants)
    if len(participants) < 3:
        raise ValueError("At least three participants are required for cue-calibrated cross-subject decoding.")
    outer_participants = _normalize_outer_participants(participants, outer_participants)

    classifier_param = decode_config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(decode_config.classifier)

    main_sets = {}
    calibration_sets = {}
    calibration_feature_config = _calibration_feature_config(calibration_config, decode_config)
    for participant in participants:
        if progress is not None:
            progress(f"LOAD main participant={participant}")
        main_sets[participant] = cross_subject.load_participant_stimulus_features(data_folder, participant, config=decode_config)
        if progress is not None:
            progress(f"LOAD cue_calibration participant={participant}")
        calibration_sets[participant] = load_participant_cue_calibration_features(data_folder, participant, config=calibration_feature_config)

    outer_rows = []
    prediction_rows = []
    for test_participant in outer_participants:
        if progress is not None:
            progress(f"START outer_test_participant={test_participant}")
        train_participants = tuple(participant for participant in participants if participant != test_participant)
        train_main_sets = [main_sets[participant] for participant in train_participants]
        train_calibration_sets = [calibration_sets[participant] for participant in train_participants]
        test_main_set = main_sets[test_participant]
        test_calibration_set = calibration_sets[test_participant]
        outer_row, participant_predictions = _evaluate_cue_calibrated_outer_fold(
            train_main_sets,
            train_calibration_sets,
            test_main_set,
            test_calibration_set,
            decode_config=decode_config,
            calibration_config=calibration_config,
            classifier_param=classifier_param,
            include_predictions=True,
            label_shuffle_seed=label_shuffle_seed if label_shuffle_control else None,
            label_shuffle_context=(int(test_participant),),
            target_calibration_label_shuffle_seed=(
                target_calibration_label_shuffle_seed if target_calibration_label_shuffle_control else None
            ),
        )
        outer_rows.append(outer_row)
        prediction_rows.extend(participant_predictions)
        if progress is not None:
            progress(f"DONE outer_test_participant={test_participant} balanced_accuracy={outer_row['balanced_accuracy']:.4f}")

    group_summary_rows = summarize_cross_subject_cue_calibrated_stimulus(
        outer_rows,
        decode_config=decode_config,
        calibration_config=calibration_config,
    )
    confusion_rows, per_stimulus_rows = cross_subject.summarize_cross_subject_predictions(prediction_rows)
    confusion_pair_rows = cross_subject.summarize_cross_subject_confusion_pairs(prediction_rows)
    return {
        "outer": outer_rows,
        "predictions": prediction_rows,
        "group_summary": group_summary_rows,
        "confusion": confusion_rows,
        "per_stimulus": per_stimulus_rows,
        "confusion_pairs": confusion_pair_rows,
    }


def export_cross_subject_cue_calibrated_stimulus(  # pylint: disable=too-many-arguments
    data_folder,
    participants,
    *,
    outer_output_path,
    group_summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    confusion_pairs_output_path=None,
    decode_config=None,
    calibration_config=None,
    outer_participants=None,
    progress=None,
    label_shuffle_control=False,
    label_shuffle_seed=0,
    target_calibration_label_shuffle_control=False,
    target_calibration_label_shuffle_seed=0,
):
    """Run cue-calibrated LOSO decoding and write compact CSV artifacts."""

    artifacts = evaluate_cross_subject_cue_calibrated_stimulus(
        data_folder,
        participants,
        decode_config=decode_config,
        calibration_config=calibration_config,
        outer_participants=outer_participants,
        progress=progress,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
        target_calibration_label_shuffle_control=target_calibration_label_shuffle_control,
        target_calibration_label_shuffle_seed=target_calibration_label_shuffle_seed,
    )
    write_alpha_metrics_csv(artifacts["outer"], outer_output_path)
    if group_summary_output_path:
        write_alpha_metrics_csv(artifacts["group_summary"], group_summary_output_path)
    if predictions_output_path:
        write_alpha_metrics_csv(artifacts["predictions"], predictions_output_path)
    if confusion_output_path:
        write_alpha_metrics_csv(artifacts["confusion"], confusion_output_path)
    if per_stimulus_output_path:
        write_alpha_metrics_csv(artifacts["per_stimulus"], per_stimulus_output_path)
    if confusion_pairs_output_path and artifacts["confusion_pairs"]:
        write_alpha_metrics_csv(artifacts["confusion_pairs"], confusion_pairs_output_path)
    return artifacts


def summarize_cross_subject_cue_calibrated_stimulus(outer_rows, *, decode_config, calibration_config):
    """Summarize cue-calibrated held-out participant scores."""

    if not outer_rows:
        return []
    base_summary_config = replace(decode_config, alignment="none")
    rows = cross_subject.summarize_cross_subject_stimulus_smoke(outer_rows, config=base_summary_config)
    for row in rows:
        fields = _cue_calibration_fields(calibration_config)
        fields["target_calibration_label_shuffle_control"] = _single_row_value(
            outer_rows,
            "target_calibration_label_shuffle_control",
            default=False,
        )
        fields["target_calibration_label_shuffle_seed"] = _single_row_value(
            outer_rows,
            "target_calibration_label_shuffle_seed",
            default="",
        )
        row.update(fields)
        row["target_calibration_participant"] = ""
        row["alignment"] = calibration_config.alignment
    return rows


def load_participant_cue_calibration_features(data_folder, participant, *, config=None):
    """Load one participant's ``Part*CueData.mat`` calibration features."""

    config = _normalized_decode_config(config or CrossSubjectStimulusConfig())
    data_path = Path(resolve_data_folder(data_folder)) / f"Part{int(participant)}CueData.mat"
    data = sio.loadmat(data_path)["data"][0]
    all_labels = cross_subject._trialinfo_labels(data)  # pylint: disable=protected-access
    trial_indices = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
        all_labels,
        config.max_trials_per_class_per_participant,
    )
    labels = all_labels[trial_indices]
    features, n_window_samples = cross_subject._extract_window_features(  # pylint: disable=protected-access
        data,
        cross_subject._centered_window(config.window_center, config.window_size),  # pylint: disable=protected-access
        feature_mode=config.feature_mode,
        trial_indices=trial_indices,
    )
    baseline_feature_mean = None
    baseline_feature_std = None
    baseline_whitening_matrix = None
    n_baseline_samples = 0
    if config.normalization in ("subject_baseline_z", "subject_baseline_whiten"):
        baseline_feature_mean, baseline_feature_std, n_baseline_samples = cross_subject._baseline_feature_statistics(  # pylint: disable=protected-access
            data,
            config,
            n_window_samples,
            trial_indices,
        )
    if config.normalization == "subject_baseline_whiten":
        baseline_whitening_matrix, n_baseline_samples = cross_subject._baseline_channel_whitening_matrix(  # pylint: disable=protected-access
            data,
            config.baseline_window,
            trial_indices,
        )
    normalized_features = cross_subject._normalize_features(  # pylint: disable=protected-access
        features,
        config,
        baseline_feature_mean,
        baseline_feature_std,
        baseline_whitening_matrix,
    )
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"Participant {participant} has {labels.shape[0]} cue labels but {features.shape[0]} feature rows.")
    return ParticipantFeatureSet(
        participant=int(participant),
        labels=labels,
        features=normalized_features,
        normalization=config.normalization,
        baseline_features=None,
        baseline_feature_mean=baseline_feature_mean,
        baseline_feature_std=baseline_feature_std,
        baseline_whitening_matrix=baseline_whitening_matrix,
        n_channels=int(cross_subject._trial_signal(data, 0).shape[0]),  # pylint: disable=protected-access
        n_window_samples=int(n_window_samples),
        n_baseline_samples=int(n_baseline_samples),
        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,
    )


def _evaluate_cue_calibrated_outer_fold(  # pylint: disable=too-many-arguments
    train_main_sets,
    train_calibration_sets,
    test_main_set,
    test_calibration_set,
    *,
    decode_config,
    calibration_config,
    classifier_param,
    include_predictions=True,
    label_shuffle_seed=None,
    label_shuffle_context=(),
    target_calibration_label_shuffle_seed=None,
):
    fitted_model, test_transform = _fit_cue_calibrated_outer_fold_model(
        train_main_sets,
        train_calibration_sets,
        test_calibration_set,
        decode_config=decode_config,
        calibration_config=calibration_config,
        classifier_param=classifier_param,
        label_shuffle_seed=label_shuffle_seed,
        label_shuffle_context=label_shuffle_context,
        target_calibration_label_shuffle_seed=target_calibration_label_shuffle_seed,
    )
    test_features = cross_subject._normalized_subject_features(test_main_set, decode_config)  # pylint: disable=protected-access
    aligned_test_features = cross_subject._apply_channel_procrustes_transform(  # pylint: disable=protected-access
        test_features,
        test_main_set,
        test_transform,
    )
    aligned_test_set = replace(test_main_set, features=aligned_test_features)
    output_config = _cue_output_decode_config(decode_config, calibration_config)
    outer_row, prediction_rows = cross_subject._score_outer_fold_model(  # pylint: disable=protected-access
        fitted_model,
        aligned_test_set,
        output_config,
        include_predictions=include_predictions,
    )
    extra_fields = _cue_calibration_fields(
        calibration_config,
        target_participant=test_main_set.participant,
        target_calibration_label_shuffle_seed=target_calibration_label_shuffle_seed,
    )
    outer_row.update(extra_fields)
    for prediction_row in prediction_rows:
        prediction_row.update(extra_fields)
    return outer_row, prediction_rows


def _fit_cue_calibrated_outer_fold_model(  # pylint: disable=too-many-arguments,too-many-locals
    train_main_sets,
    train_calibration_sets,
    test_calibration_set,
    *,
    decode_config,
    calibration_config,
    classifier_param,
    label_shuffle_seed=None,
    label_shuffle_context=(),
    target_calibration_label_shuffle_seed=None,
):
    if len(train_main_sets) != len(train_calibration_sets):
        raise ValueError("Training main and cue-calibration sets must have the same length.")
    if not train_main_sets:
        raise ValueError("At least one training participant is required.")

    source_label_arrays = [np.asarray(feature_set.labels, dtype=int) for feature_set in train_calibration_sets]
    target_labels = np.asarray(test_calibration_set.labels, dtype=int)
    if target_calibration_label_shuffle_seed is not None:
        target_labels = _permuted_labels(
            target_labels,
            seed=target_calibration_label_shuffle_seed,
            context=(int(test_calibration_set.participant),),
        )
    common_classes = _source_target_common_classes(source_label_arrays, target_labels)
    if len(common_classes) < 2:
        raise ValueError("Cue calibration requires at least two stimulus classes shared by source and target participants.")

    source_class_patterns = [
        cross_subject._participant_class_channel_patterns(  # pylint: disable=protected-access
            cross_subject._normalized_subject_features(calibration_set, _calibration_feature_config(calibration_config, decode_config)),  # pylint: disable=protected-access
            labels,
            calibration_set,
            common_classes,
        )
        for calibration_set, labels in zip(train_calibration_sets, source_label_arrays)
    ]
    source_template, source_transforms = _source_only_procrustes_template_and_transforms(source_class_patterns)
    target_patterns = cross_subject._participant_class_channel_patterns(  # pylint: disable=protected-access
        cross_subject._normalized_subject_features(test_calibration_set, _calibration_feature_config(calibration_config, decode_config)),  # pylint: disable=protected-access
        target_labels,
        test_calibration_set,
        common_classes,
    )
    target_transform = cross_subject._channel_procrustes_transform(target_patterns, source_template)  # pylint: disable=protected-access

    train_features_by_subject = [
        cross_subject._normalized_subject_features(feature_set, decode_config)  # pylint: disable=protected-access
        for feature_set in train_main_sets
    ]
    aligned_train_features_by_subject = [
        cross_subject._apply_channel_procrustes_transform(features, feature_set, transform)  # pylint: disable=protected-access
        for feature_set, features, transform in zip(train_main_sets, train_features_by_subject, source_transforms)
    ]
    train_label_arrays = [
        cross_subject._training_labels(  # pylint: disable=protected-access
            feature_set,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=label_shuffle_context,
        )
        for feature_set in train_main_sets
    ]
    train_features = np.vstack(aligned_train_features_by_subject)
    train_labels_one_based = np.concatenate(train_label_arrays)
    train_labels = train_labels_one_based - 1
    feature_transform_metadata = None
    fit_training_feature_transform = getattr(cross_subject, "_fit_training_feature_transform", None)
    if fit_training_feature_transform is not None:
        train_features, feature_transform_metadata = fit_training_feature_transform(train_features, train_main_sets, decode_config)
    train_window = cross_subject._centered_window(decode_config.window_center, decode_config.window_size)  # pylint: disable=protected-access
    model_bundle = fit_reptrace_window_model(
        train_features,
        train_labels,
        fit_model=lambda features, labels: train_multiclass_classifier(
            features,
            labels,
            decode_config.classifier,
            classifier_param,
            random_state=decode_config.random_state,
        ),
        components_pca=decode_config.components_pca,
        train_window=train_window,
    )
    fitted_model = {
        "classifier_param": classifier_param,
        "model_bundle": model_bundle,
        "n_train_participants": len(train_main_sets),
        "train_class_counts": Counter(train_labels_one_based.tolist()),
        "train_labels": train_labels,
        "train_participants": tuple(feature_set.participant for feature_set in train_main_sets),
        "train_window": train_window,
        "label_shuffle_control": label_shuffle_seed is not None,
        "label_shuffle_seed": "" if label_shuffle_seed is None else int(label_shuffle_seed),
        "alignment_metadata": cross_subject._alignment_metadata(  # pylint: disable=protected-access
            calibration_config.alignment,
            common_classes=common_classes,
            aligned_participants=tuple(feature_set.participant for feature_set in train_main_sets) + (test_calibration_set.participant,),
        ),
    }
    if feature_transform_metadata is not None:
        fitted_model["feature_transform_metadata"] = feature_transform_metadata
    return fitted_model, target_transform


def _source_only_procrustes_template_and_transforms(source_class_patterns):
    if not source_class_patterns:
        raise ValueError("At least one source participant is required for cue calibration.")
    template = np.asarray(source_class_patterns[0], dtype=float)
    transforms = []
    for _ in range(3):
        transforms = [
            cross_subject._channel_procrustes_transform(patterns, template)  # pylint: disable=protected-access
            for patterns in source_class_patterns
        ]
        aligned_patterns = [
            cross_subject._apply_channel_pattern_transform(patterns, transform)  # pylint: disable=protected-access
            for patterns, transform in zip(source_class_patterns, transforms)
        ]
        template = np.mean(np.stack(aligned_patterns, axis=0), axis=0)
    transforms = [
        cross_subject._channel_procrustes_transform(patterns, template)  # pylint: disable=protected-access
        for patterns in source_class_patterns
    ]
    return template, transforms


def _source_target_common_classes(source_label_arrays, target_labels):
    label_sets = [set(np.asarray(labels, dtype=int).tolist()) for labels in source_label_arrays]
    label_sets.append(set(np.asarray(target_labels, dtype=int).tolist()))
    if not label_sets:
        return tuple()
    return tuple(sorted(set.intersection(*label_sets)))


def _permuted_labels(labels, *, seed, context):
    labels = np.asarray(labels, dtype=int)
    seed_values = [int(seed), *[int(value) for value in context]]
    rng = np.random.default_rng(np.random.SeedSequence(seed_values))
    return rng.permutation(labels)


def _normalized_decode_config(config):
    config = cross_subject._normalized_config(config)  # pylint: disable=protected-access
    if config.alignment != "none":
        raise ValueError("Cue-calibrated decoding expects decode_config.alignment='none'; calibration supplies the alignment.")
    return config


def _normalized_cue_calibration_config(config, decode_config):
    calibration_data = str(config.calibration_data).strip().lower().replace("-", "_")
    if calibration_data not in CUE_CALIBRATION_DATASETS:
        raise ValueError(f"calibration_data must be one of {CUE_CALIBRATION_DATASETS}.")
    alignment = str(config.alignment).strip().lower().replace("-", "_")
    if alignment not in CUE_CALIBRATION_ALIGNMENTS:
        raise ValueError(f"calibration alignment must be one of {CUE_CALIBRATION_ALIGNMENTS}.")
    template_policy = str(config.template_policy).strip().lower().replace("-", "_")
    if template_policy not in CUE_CALIBRATION_TEMPLATE_POLICIES:
        raise ValueError(f"template_policy must be one of {CUE_CALIBRATION_TEMPLATE_POLICIES}.")
    feature_mode = _decode_reference_or_token(config.feature_mode, decode_config.feature_mode, cross_subject._normalize_feature_mode)  # pylint: disable=protected-access
    normalization = _decode_reference_or_token(config.normalization, decode_config.normalization, cross_subject._normalize_normalization)  # pylint: disable=protected-access
    return CueCalibrationConfig(
        calibration_data=calibration_data,
        alignment=alignment,
        template_policy=template_policy,
        window_center=float(config.window_center),
        window_size=float(config.window_size),
        baseline_window=(float(config.baseline_window[0]), float(config.baseline_window[1])),
        feature_mode=feature_mode,
        normalization=normalization,
        max_trials_per_class_per_participant=cross_subject._normalize_trial_cap(  # pylint: disable=protected-access
            config.max_trials_per_class_per_participant
        ),
    )


def _decode_reference_or_token(value, decode_value, normalizer):
    token = str(value).strip().lower().replace("-", "_")
    if token == DECODE_REFERENCE_TOKEN:
        return decode_value
    return normalizer(token)


def _calibration_feature_config(calibration_config, decode_config):
    return CrossSubjectStimulusConfig(
        window_center=calibration_config.window_center,
        window_size=calibration_config.window_size,
        baseline_window=calibration_config.baseline_window,
        feature_mode=calibration_config.feature_mode,
        normalization=calibration_config.normalization,
        alignment="none",
        classifier=decode_config.classifier,
        classifier_param=decode_config.classifier_param,
        components_pca=decode_config.components_pca,
        max_trials_per_class_per_participant=calibration_config.max_trials_per_class_per_participant,
        chance_classes=decode_config.chance_classes,
        random_state=decode_config.random_state,
        signflip_permutations=decode_config.signflip_permutations,
        signflip_seed=decode_config.signflip_seed,
    )


def _cue_output_decode_config(decode_config, calibration_config):
    del calibration_config
    return replace(decode_config, alignment="none")


def _cue_calibration_fields(calibration_config, *, target_participant="", target_calibration_label_shuffle_seed=None):
    return {
        "alignment": calibration_config.alignment,
        "alignment_test_transform": "cue_target_transform",
        "alignment_target_centering": "cue_supervised_localizer",
        "calibration_data": calibration_config.calibration_data,
        "calibration_alignment": calibration_config.alignment,
        "calibration_template_policy": calibration_config.template_policy,
        "calibration_window_center_s": calibration_config.window_center,
        "calibration_window_size_s": calibration_config.window_size,
        "calibration_window_start_s": cross_subject._centered_window(  # pylint: disable=protected-access
            calibration_config.window_center,
            calibration_config.window_size,
        )[0],
        "calibration_window_stop_s": cross_subject._centered_window(  # pylint: disable=protected-access
            calibration_config.window_center,
            calibration_config.window_size,
        )[1],
        "calibration_baseline_window_start_s": calibration_config.baseline_window[0],
        "calibration_baseline_window_stop_s": calibration_config.baseline_window[1],
        "calibration_feature_mode": calibration_config.feature_mode,
        "calibration_normalization": calibration_config.normalization,
        "calibration_max_trials_per_class_per_participant": calibration_config.max_trials_per_class_per_participant,
        "target_calibration_participant": target_participant,
        "target_calibration_label_shuffle_control": target_calibration_label_shuffle_seed is not None,
        "target_calibration_label_shuffle_seed": "" if target_calibration_label_shuffle_seed is None else int(target_calibration_label_shuffle_seed),
    }


def _normalize_outer_participants(participants, outer_participants):
    if outer_participants is None:
        return tuple(participants)
    outer_participants = tuple(int(participant) for participant in outer_participants)
    if not outer_participants:
        raise ValueError("At least one outer participant is required.")
    unknown = sorted(set(outer_participants) - set(participants))
    if unknown:
        raise ValueError(f"Outer participants must be part of participants: {unknown}")
    return outer_participants


def _single_row_value(rows, key, *, default=""):
    values = []
    for row in rows:
        value = row.get(key, default)
        if value in (None, ""):
            continue
        if value not in values:
            values.append(value)
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return ";".join(str(value) for value in values)
