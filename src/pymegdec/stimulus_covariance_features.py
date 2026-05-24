"""Compatibility bridge for covariance-feature BUSH-MEG stimulus decoding.

The reusable covariance LOSO implementation now lives in
:mod:`neureptrace.bushmeg_covariance_loso`.  PyMEGDec keeps the historical
``pymegdec stimulus cross-subject-covariance`` command as a thin translator from
legacy command-line arguments to a NeuRepTrace dataset/workflow config.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neureptrace import bushmeg_covariance_loso as _nrt_covariance
from neureptrace.bushmeg_covariance_loso import (
    COVARIANCE_FEATURE_MODES,
    DEFAULT_COVARIANCE_EPSILON,
    DEFAULT_COVARIANCE_FEATURE_MODE,
    DEFAULT_COVARIANCE_MAX_CHANNELS,
    DEFAULT_COVARIANCE_SHRINKAGE,
    CovarianceCandidateSpec,
    CovarianceWindow,
    covariance_feature_vector,
    normalize_covariance_feature_mode,
)

from pymegdec.cli import normalize_argv
from pymegdec.data_config import resolve_data_folder

DEFAULT_COVARIANCE_TIME_WINDOWS = ((0.05, 0.30),)
DEFAULT_COVARIANCE_BASELINE_WINDOW = (-0.35, -0.05)
DEFAULT_COVARIANCE_NORMALIZATION = "subject_baseline_whiten"
DEFAULT_COVARIANCE_PROJECTION = "pca"
DEFAULT_COVARIANCE_COMPONENTS = (32, 64, 128)
DEFAULT_COVARIANCE_CLASSIFIER = "multinomial-logistic"
DEFAULT_COVARIANCE_CLASSIFIER_PARAMS = (0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_COVARIANCE_PARTICIPANTS = "1-4,6,8,9,10,13-27"
COVARIANCE_FEATURE_FAMILY = "covariance"


# Backward-compatible aliases for code that imported PyMEGDec's former private helpers.
_normalize_covariance_feature_mode = normalize_covariance_feature_mode
_covariance_feature_vector = covariance_feature_vector


def _token_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(token.strip() for token in value.split(",") if token.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _float_list(value: Any) -> tuple[float, ...]:
    values = tuple(float(token) for token in _token_list(value)) if isinstance(value, str) else tuple(float(item) for item in value)
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("Numeric grid values must be finite.")
    return values


def _classifier_param_list(value: Any) -> tuple[float, ...]:
    params: list[float] = []
    for token in _token_list(value):
        lowered = token.lower()
        if lowered in {"default", "nan"}:
            raise argparse.ArgumentTypeError(
                "The NeuRepTrace covariance wrapper needs explicit classifier parameters; use values such as 0.03,0.1,1.0."
            )
        params.append(float(token))
    if not params:
        raise argparse.ArgumentTypeError("At least one classifier parameter is required.")
    return tuple(params)


def _component_list(value: Any) -> tuple[int | None, ...]:
    components: list[int | None] = []
    for token in _token_list(value):
        lowered = token.lower()
        if lowered in {"inf", "infinity", "none", "null"}:
            components.append(None)
        else:
            parsed = int(token)
            if parsed < 1:
                raise argparse.ArgumentTypeError("Component counts must be positive integers, inf, or none.")
            components.append(parsed)
    if not components:
        raise argparse.ArgumentTypeError("At least one component value is required.")
    return tuple(components)


def _parse_time_window(value: str | Sequence[float]) -> tuple[float, float]:
    if isinstance(value, str):
        start_text, stop_text = value.split(":", maxsplit=1)
        start, stop = float(start_text), float(stop_text)
    else:
        start, stop = map(float, value)
    if not np.all(np.isfinite([start, stop])) or stop <= start:
        raise argparse.ArgumentTypeError("Time windows must be finite start:stop pairs with stop > start.")
    return start, stop


def _parse_time_windows(value: Any) -> tuple[tuple[float, float], ...]:
    if isinstance(value, str):
        windows = tuple(_parse_time_window(token.strip()) for token in value.split(",") if token.strip())
    else:
        windows = tuple(_parse_time_window(item) for item in value)
    if not windows:
        raise argparse.ArgumentTypeError("At least one time window is required.")
    return windows


def _single_value(name: str, values: Sequence[Any]) -> Any:
    values = tuple(values)
    if len(values) != 1:
        raise ValueError(
            f"The NeuRepTrace covariance workflow supports one {name} per run. "
            f"Run the command multiple times or use a NeuRepTrace config for multiple {name} variants."
        )
    return values[0]


def _first_value(values: Sequence[Any], default: Any) -> Any:
    values = tuple(values)
    return values[0] if values else default


def _window_name(index: int, start: float, stop: float) -> str:
    start_ms = int(round(1000.0 * start))
    stop_ms = int(round(1000.0 * stop))
    return f"cov_{index:02d}_{start_ms:03d}_{stop_ms:03d}ms"


def _window_specs(time_windows: Sequence[tuple[float, float]]) -> list[dict[str, float | str]]:
    return [
        {"name": _window_name(index, start, stop), "start": float(start), "stop": float(stop)}
        for index, (start, stop) in enumerate(time_windows)
    ]


def _participant_config_value(participants: str | Sequence[int | str]) -> str | list[int | str]:
    if isinstance(participants, str):
        return participants
    return list(participants)


def _max_window_stop(time_windows: Sequence[tuple[float, float]], baseline_window: tuple[float, float]) -> float:
    return max([baseline_window[1], *[window[1] for window in time_windows]])


def _min_window_start(time_windows: Sequence[tuple[float, float]], baseline_window: tuple[float, float]) -> float:
    return min([baseline_window[0], *[window[0] for window in time_windows]])


def build_neureptrace_covariance_config(  # pylint: disable=too-many-arguments
    *,
    data_folder: str | Path,
    participants: str | Sequence[int | str] = DEFAULT_COVARIANCE_PARTICIPANTS,
    time_windows: Sequence[tuple[float, float]] = DEFAULT_COVARIANCE_TIME_WINDOWS,
    baseline_window: tuple[float, float] = DEFAULT_COVARIANCE_BASELINE_WINDOW,
    normalizations: Sequence[str] = (DEFAULT_COVARIANCE_NORMALIZATION,),
    feature_modes: Sequence[str] = (DEFAULT_COVARIANCE_FEATURE_MODE,),
    covariance_shrinkages: Sequence[float] = (DEFAULT_COVARIANCE_SHRINKAGE,),
    covariance_epsilons: Sequence[float] = (DEFAULT_COVARIANCE_EPSILON,),
    covariance_max_channels: Sequence[int] = (DEFAULT_COVARIANCE_MAX_CHANNELS,),
    projections: Sequence[str] = (DEFAULT_COVARIANCE_PROJECTION,),
    classifiers: Sequence[str] = (DEFAULT_COVARIANCE_CLASSIFIER,),
    classifier_params: Sequence[float] = DEFAULT_COVARIANCE_CLASSIFIER_PARAMS,
    components_values: Sequence[int | None] = DEFAULT_COVARIANCE_COMPONENTS,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 0,
    random_state: int | None = None,
    max_iter: int = 2500,
) -> dict[str, Any]:
    """Build a NeuRepTrace covariance-LOSO config from legacy PyMEGDec arguments."""

    time_windows = tuple(_parse_time_window(window) for window in time_windows)
    baseline_window = _parse_time_window(baseline_window)
    normalization = str(_single_value("normalization", tuple(normalizations)))
    if random_state not in {None, 0}:
        warnings.warn(
            "NeuRepTrace covariance LOSO uses its own deterministic workflow seed; the legacy random-state argument is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "schema_version": "neureptrace.dataset.v1",
        "paths": {"base": "cwd"},
        "dataset": {
            "name": "bush_meg",
            "type": "fieldtrip_mat",
            "root": str(data_folder),
            "participant_file": "Part{participant}Data.mat",
            "variable": "data",
            "channel_type": "mag",
            "fields": {
                "trial": "trial",
                "time": "time",
                "label": "label",
                "trialinfo": "trialinfo",
                "sensor_geometry": "grad",
            },
        },
        "participants": {"ids": _participant_config_value(participants)},
        "validation": {"trim_channel_labels_to_data": True, "channel_policy": "exact"},
        "metadata": {
            "columns": [
                {"name": "stimulus_class", "index": 0},
                {"name": "condition", "index": 1, "optional": True},
            ]
        },
        "preprocessing": {
            "tmin": float(_min_window_start(time_windows, baseline_window)),
            "tmax": float(_max_window_stop(time_windows, baseline_window)),
            "normalization": normalization,
            "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
        },
        "decoding": {
            "label_column": "stimulus_class",
            "group_column": "participant",
            "classifier": str(_first_value(classifiers, DEFAULT_COVARIANCE_CLASSIFIER)),
            "emission_mode": "uncalibrated",
            "feature_preprocessor": str(_first_value(projections, DEFAULT_COVARIANCE_PROJECTION)),
            "pca_components": None if components_values[0] is None else int(components_values[0]),
            "max_iter": int(max_iter),
        },
        "covariance_loso": {
            "group_column": "participant",
            "selection_metric": "balanced_accuracy",
            "label_shuffle_control": bool(label_shuffle_control),
            "label_shuffle_seed": int(label_shuffle_seed),
            "candidate_grid": {
                "time_windows": _window_specs(time_windows),
                "feature_modes": [normalize_covariance_feature_mode(mode) for mode in feature_modes],
                "covariance_shrinkages": [float(value) for value in covariance_shrinkages],
                "covariance_epsilons": [float(value) for value in covariance_epsilons],
                "covariance_max_channels": [int(value) for value in covariance_max_channels],
                "decoders": [str(value) for value in classifiers],
                "emission_modes": ["uncalibrated"],
                "feature_preprocessors": [str(value) for value in projections],
                "pca_components": [None if value is None else int(value) for value in components_values],
                "c_grid": [float(value) for value in classifier_params],
            },
        },
        "outputs": {
            "base_dir": "outputs/neureptrace_covariance_loso",
            "covariance_loso_summary_csv": "covariance_loso_summary.csv",
            "covariance_loso_inner_cv_csv": "covariance_loso_inner_cv.csv",
            "covariance_loso_predictions_csv": "covariance_loso_predictions.csv",
            "provenance": True,
            "hash_input_files": True,
        },
    }


def write_neureptrace_covariance_config(config: dict[str, Any], path: str | Path) -> Path:
    """Write a generated NeuRepTrace covariance config as JSON and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def make_covariance_candidate_configs(  # pylint: disable=too-many-arguments
    *,
    time_windows=DEFAULT_COVARIANCE_TIME_WINDOWS,
    covariance_feature_modes=(DEFAULT_COVARIANCE_FEATURE_MODE,),
    covariance_shrinkages=(DEFAULT_COVARIANCE_SHRINKAGE,),
    covariance_epsilons=(DEFAULT_COVARIANCE_EPSILON,),
    covariance_max_channels=(DEFAULT_COVARIANCE_MAX_CHANNELS,),
    projections=(DEFAULT_COVARIANCE_PROJECTION,),
    classifiers=(DEFAULT_COVARIANCE_CLASSIFIER,),
    classifier_params=DEFAULT_COVARIANCE_CLASSIFIER_PARAMS,
    components_values=DEFAULT_COVARIANCE_COMPONENTS,
    **_legacy_kwargs,
) -> tuple[CovarianceCandidateSpec, ...]:
    """Return NeuRepTrace covariance candidate specs for compatibility tests/scripts."""

    candidates: list[CovarianceCandidateSpec] = []
    windows = [CovarianceWindow(name=_window_name(index, *window), start=window[0], stop=window[1]) for index, window in enumerate(_parse_time_windows(time_windows))]
    for window in windows:
        for mode in covariance_feature_modes:
            for shrinkage in covariance_shrinkages:
                for epsilon in covariance_epsilons:
                    for max_channels in covariance_max_channels:
                        for decoder in classifiers:
                            for projection in projections:
                                for components in components_values:
                                    for classifier_param in classifier_params:
                                        normalized_mode = normalize_covariance_feature_mode(mode)
                                        components_value = None if components is None or components == float("inf") else int(components)
                                        name = "__".join(
                                            [
                                                window.name,
                                                normalized_mode,
                                                f"shrink{float(shrinkage):g}",
                                                f"eps{float(epsilon):g}",
                                                f"covch{int(max_channels)}",
                                                str(decoder).replace("-", "_"),
                                                str(projection).replace("-", "_"),
                                                "pca" + ("none" if components_value is None else str(components_value)),
                                                f"c{float(classifier_param):g}",
                                            ]
                                        )
                                        candidates.append(
                                            CovarianceCandidateSpec(
                                                name=name,
                                                decoder=str(decoder),
                                                emission_mode="uncalibrated",
                                                feature_preprocessor=str(projection),
                                                pca_components=components_value,
                                                classifier_param=float(classifier_param),
                                                window=window,
                                                covariance_feature_mode=normalized_mode,
                                                covariance_shrinkage=float(shrinkage),
                                                covariance_epsilon=float(epsilon),
                                                covariance_max_channels=int(max_channels),
                                            )
                                        )
    return tuple(candidates)


def _write_group_summary(summary: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"n_subjects": int(len(summary)), "feature_family": COVARIANCE_FEATURE_FAMILY}
    for column in ("accuracy", "balanced_accuracy", "top2_accuracy", "top3_accuracy", "log_loss", "brier_score"):
        if column in summary.columns:
            values = pd.to_numeric(summary[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    for column in ("candidate", "covariance_feature_mode", "covariance_shrinkage", "covariance_epsilon", "feature_preprocessor", "pca_components"):
        if column in summary.columns:
            row[f"selected_{column}_counts"] = "|".join(f"{key}:{count}" for key, count in summary[column].astype(str).value_counts().sort_index().items())
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_selected(summary: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_columns = [
        column
        for column in (
            "outer_test_subject",
            "candidate",
            "inner_selection_metric",
            "inner_mean_score",
            "inner_std_score",
            "window_name",
            "window_start",
            "window_stop",
            "covariance_feature_mode",
            "covariance_shrinkage",
            "covariance_epsilon",
            "covariance_max_channels",
            "decoder",
            "feature_preprocessor",
            "pca_components",
            "classifier_param",
        )
        if column in summary.columns
    ]
    summary[selected_columns].to_csv(path, index=False)


def _write_prediction_derivatives(
    predictions_path: str | Path | None,
    *,
    confusion_output: str | Path | None = None,
    per_stimulus_output: str | Path | None = None,
    confusion_pairs_output: str | Path | None = None,
) -> None:
    if predictions_path is None or not Path(predictions_path).exists():
        return
    predictions = pd.read_csv(predictions_path)
    if not {"true_label", "predicted_label"}.issubset(predictions.columns):
        return
    if confusion_output is not None:
        confusion_path = Path(confusion_output)
        confusion_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.groupby(["true_label", "predicted_label"], dropna=False).size().reset_index(name="count").to_csv(confusion_path, index=False)
    if per_stimulus_output is not None:
        per_path = Path(per_stimulus_output)
        per_path.parent.mkdir(parents=True, exist_ok=True)
        per = predictions.assign(is_correct=predictions["true_label"] == predictions["predicted_label"])
        per.groupby("true_label", dropna=False).agg(n_trials=("true_label", "size"), n_correct=("is_correct", "sum"), accuracy=("is_correct", "mean")).reset_index().to_csv(per_path, index=False)
    if confusion_pairs_output is not None:
        pair_path = Path(confusion_pairs_output)
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        errors = predictions.loc[predictions["true_label"] != predictions["predicted_label"], ["true_label", "predicted_label"]].copy()
        if errors.empty:
            pd.DataFrame(columns=["stimulus_a", "stimulus_b", "count"]).to_csv(pair_path, index=False)
        else:
            errors["stimulus_a"] = errors[["true_label", "predicted_label"]].min(axis=1)
            errors["stimulus_b"] = errors[["true_label", "predicted_label"]].max(axis=1)
            errors.groupby(["stimulus_a", "stimulus_b"], dropna=False).size().reset_index(name="count").to_csv(pair_path, index=False)


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run PyMEGDec's legacy covariance command through the NeuRepTrace covariance LOSO workflow.",
    )
    parser.add_argument("--neureptrace-config", type=Path, help="Run an existing NeuRepTrace covariance LOSO config instead of generating one from legacy options.")
    parser.add_argument("--write-neureptrace-config", type=Path, help="Write the generated NeuRepTrace config to this JSON path before running.")
    parser.add_argument("--data-dir", dest="data_folder", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participants", default=DEFAULT_COVARIANCE_PARTICIPANTS, help="Participant ids such as 1-4,6,8.")
    parser.add_argument("--outer-participants", default=None, help="Unsupported by the NeuRepTrace-backed wrapper; use a config with a participant subset instead.")
    parser.add_argument("--time-windows", default="0.05:0.30", help="Comma-separated covariance crop windows as start:stop pairs.")
    parser.add_argument("--baseline-window", default="-0.35:-0.05", help="Baseline window as start:stop in seconds.")
    parser.add_argument("--normalizations", default=DEFAULT_COVARIANCE_NORMALIZATION, help="Single subject normalization mode. Multiple values are not supported by this wrapper.")
    parser.add_argument("--feature-modes", default=DEFAULT_COVARIANCE_FEATURE_MODE, help="Comma-separated covariance feature modes.")
    parser.add_argument("--covariance-shrinkages", default=str(DEFAULT_COVARIANCE_SHRINKAGE), help="Comma-separated shrinkage values in [0,1].")
    parser.add_argument("--covariance-epsilons", default=str(DEFAULT_COVARIANCE_EPSILON), help="Comma-separated positive eigenvalue floors.")
    parser.add_argument("--covariance-max-channels", default=str(DEFAULT_COVARIANCE_MAX_CHANNELS), help="Comma-separated channel caps passed to NeuRepTrace.")
    parser.add_argument("--projections", default=DEFAULT_COVARIANCE_PROJECTION, help="Comma-separated NeuRepTrace feature preprocessors such as pca, none, pca_whiten, pls_da.")
    parser.add_argument("--classifiers", default=DEFAULT_COVARIANCE_CLASSIFIER, help="Comma-separated NeuRepTrace decoder names.")
    parser.add_argument("--classifier-params", default=",".join(str(value) for value in DEFAULT_COVARIANCE_CLASSIFIER_PARAMS), help="Comma-separated classifier C/grid values.")
    parser.add_argument("--components-values", default=",".join(str(value) for value in DEFAULT_COVARIANCE_COMPONENTS), help="Comma-separated PCA component counts, none, or inf.")
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Accepted for legacy compatibility but handled only by native NeuRepTrace configs.")
    parser.add_argument("--trial-selection", default=None, help="Accepted for legacy compatibility but handled only by native NeuRepTrace configs.")
    parser.add_argument("--trial-selection-seed", type=int, default=None, help="Accepted for legacy compatibility but handled only by native NeuRepTrace configs.")
    parser.add_argument("--chance-classes", type=int, default=16, help="Accepted for legacy compatibility; NeuRepTrace infers classes from labels.")
    parser.add_argument("--random-state", type=int, default=0, help="Accepted for compatibility; NeuRepTrace uses its workflow seed.")
    parser.add_argument("--label-shuffle-control", action="store_true", help="Shuffle training labels within each source fold.")
    parser.add_argument("--label-shuffle-seed", type=int, default=0, help="Seed for the nested label-shuffle control.")
    parser.add_argument("--signflip-permutations", type=int, default=10000, help="Accepted for compatibility; the NeuRepTrace summary reports per-subject rows.")
    parser.add_argument("--signflip-seed", type=int, default=0, help="Accepted for compatibility.")
    parser.add_argument("--max-iter", type=int, default=2500, help="Maximum classifier iterations in the NeuRepTrace workflow.")
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_covariance_outer.csv", help="NeuRepTrace outer-subject summary CSV.")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_covariance_group_summary.csv", help="Small PyMEGDec compatibility aggregate summary CSV.")
    parser.add_argument("--inner-validation-output", default="outputs/stimulus_cross_subject_covariance_inner_validation.csv", help="NeuRepTrace inner LOSO candidate-score CSV.")
    parser.add_argument("--selected-output", default="outputs/stimulus_cross_subject_covariance_selected.csv", help="Small PyMEGDec compatibility selected-candidate CSV.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_covariance_predictions.csv", help="NeuRepTrace held-out trial probability CSV.")
    parser.add_argument("--confusion-output", default="outputs/stimulus_cross_subject_covariance_confusion.csv", help="Derived compatibility confusion-count CSV.")
    parser.add_argument("--per-stimulus-output", default="outputs/stimulus_cross_subject_covariance_per_stimulus.csv", help="Derived compatibility per-stimulus CSV.")
    parser.add_argument("--confusion-pairs-output", default="outputs/stimulus_cross_subject_covariance_confusion_pairs.csv", help="Derived compatibility confusion-pair CSV.")
    return parser


def _generated_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    unsupported = []
    if args.outer_participants:
        unsupported.append("--outer-participants")
    if args.max_trials_per_class_per_participant is not None:
        unsupported.append("--max-trials-per-class-per-participant")
    if args.trial_selection is not None:
        unsupported.append("--trial-selection")
    if args.trial_selection_seed is not None:
        unsupported.append("--trial-selection-seed")
    if unsupported:
        raise ValueError(
            "The NeuRepTrace-backed PyMEGDec wrapper cannot translate these legacy options: "
            + ", ".join(unsupported)
            + ". Use a native NeuRepTrace covariance_loso config for this run."
        )

    data_folder = resolve_data_folder(args.data_folder)
    return build_neureptrace_covariance_config(
        data_folder=data_folder,
        participants=args.participants,
        time_windows=_parse_time_windows(args.time_windows),
        baseline_window=_parse_time_window(args.baseline_window),
        normalizations=_token_list(args.normalizations),
        feature_modes=_token_list(args.feature_modes),
        covariance_shrinkages=_float_list(args.covariance_shrinkages),
        covariance_epsilons=_float_list(args.covariance_epsilons),
        covariance_max_channels=tuple(int(value) for value in _float_list(args.covariance_max_channels)),
        projections=_token_list(args.projections),
        classifiers=_token_list(args.classifiers),
        classifier_params=_classifier_param_list(args.classifier_params),
        components_values=_component_list(args.components_values),
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )


def _run_neureptrace_config(config_path: Path, args: argparse.Namespace) -> pd.DataFrame:
    return _nrt_covariance.run_bushmeg_covariance_loso(
        config_path,
        out_path=args.outer_output,
        inner_cv_out_path=args.inner_validation_output,
        predictions_out_path=args.predictions_output,
    )


def stimulus_cross_subject_covariance(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    try:
        if args.neureptrace_config is not None:
            config_path = Path(args.neureptrace_config)
            summary = _run_neureptrace_config(config_path, args)
        else:
            config = _generated_config_from_args(args)
            if args.write_neureptrace_config is not None:
                config_path = write_neureptrace_covariance_config(config, args.write_neureptrace_config)
                summary = _run_neureptrace_config(config_path, args)
            else:
                with tempfile.TemporaryDirectory(prefix="pymegdec-neureptrace-covariance-") as tmp_dir:
                    config_path = write_neureptrace_covariance_config(config, Path(tmp_dir) / "covariance_loso.json")
                    summary = _run_neureptrace_config(config_path, args)
    except ValueError as exc:
        parser.error(str(exc))

    _write_group_summary(summary, args.summary_output)
    _write_selected(summary, args.selected_output)
    _write_prediction_derivatives(
        args.predictions_output,
        confusion_output=args.confusion_output,
        per_stimulus_output=args.per_stimulus_output,
        confusion_pairs_output=args.confusion_pairs_output,
    )
    print(f"Wrote {len(summary)} NeuRepTrace covariance LOSO outer rows to {args.outer_output}")
    print(f"Wrote compatibility group summary to {args.summary_output}")
    print(f"Wrote inner validation rows to {args.inner_validation_output}")
    print(f"Wrote prediction rows to {args.predictions_output}")
    return 0


def export_nested_covariance_stimulus(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Deprecated placeholder for the removed PyMEGDec-native covariance implementation."""

    raise RuntimeError(
        "PyMEGDec no longer owns covariance-feature nested LOSO. Use "
        "neureptrace.bushmeg_covariance_loso.run_bushmeg_covariance_loso or the "
        "pymegdec stimulus cross-subject-covariance compatibility CLI."
    )


def evaluate_nested_covariance_stimulus(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Deprecated placeholder for the removed PyMEGDec-native covariance implementation."""

    return export_nested_covariance_stimulus(*_args, **_kwargs)


def load_participant_covariance_features(*_args: Any, **_kwargs: Any) -> Any:
    """Deprecated placeholder for the removed PyMEGDec-native feature loader."""

    raise RuntimeError(
        "PyMEGDec no longer loads covariance features itself. Use NeuRepTrace's "
        "CovarianceFeatureCache and covariance_feature_vector helpers instead."
    )


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return stimulus_cross_subject_covariance(argv, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COVARIANCE_FEATURE_FAMILY",
    "COVARIANCE_FEATURE_MODES",
    "CovarianceCandidateSpec",
    "CovarianceWindow",
    "DEFAULT_COVARIANCE_BASELINE_WINDOW",
    "DEFAULT_COVARIANCE_CLASSIFIER",
    "DEFAULT_COVARIANCE_CLASSIFIER_PARAMS",
    "DEFAULT_COVARIANCE_COMPONENTS",
    "DEFAULT_COVARIANCE_EPSILON",
    "DEFAULT_COVARIANCE_FEATURE_MODE",
    "DEFAULT_COVARIANCE_NORMALIZATION",
    "DEFAULT_COVARIANCE_PARTICIPANTS",
    "DEFAULT_COVARIANCE_PROJECTION",
    "DEFAULT_COVARIANCE_SHRINKAGE",
    "DEFAULT_COVARIANCE_TIME_WINDOWS",
    "build_neureptrace_covariance_config",
    "covariance_feature_vector",
    "evaluate_nested_covariance_stimulus",
    "export_nested_covariance_stimulus",
    "load_participant_covariance_features",
    "make_covariance_candidate_configs",
    "normalize_covariance_feature_mode",
    "stimulus_cross_subject_covariance",
    "write_neureptrace_covariance_config",
]
