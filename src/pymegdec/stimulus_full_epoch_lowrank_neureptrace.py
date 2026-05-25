"""Compatibility CLI for NeuRepTrace BUSH-MEG supervised-lowrank LOSO."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.bushmeg_supervised_lowrank_loso import run_supervised_lowrank_loso

from pymegdec.cli import normalize_argv
from pymegdec.data_config import resolve_data_folder
from pymegdec.reaction_time_analysis import parse_participant_spec

DEFAULT_PARTICIPANTS = "1-4,6,8,9,10,13-27"


def _token_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(token).strip() for token in value if str(token).strip()]


def _float_list(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        return [float(token.strip()) for token in value.split(",") if token.strip()]
    return [float(token) for token in value]


def _int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(float(token.strip())) for token in value.split(",") if token.strip()]
    return [int(token) for token in value]


def _time_window(text: str | Sequence[float]) -> tuple[float, float]:
    if isinstance(text, str):
        parts = _float_list(text.replace(":", ","))
    else:
        parts = [float(value) for value in text]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Time window must have start,stop or start:stop.")
    if parts[1] <= parts[0]:
        raise argparse.ArgumentTypeError("Time-window stop must be greater than start.")
    return parts[0], parts[1]


def _window_name(index: int, window: tuple[float, float]) -> str:
    start, stop = window
    return f"epoch_{index:02d}_{int(round(1000 * start)):04d}_{int(round(1000 * stop)):04d}ms".replace("-", "m")


def _delta_values(temporal_feature_modes: Sequence[str]) -> list[bool]:
    values = []
    for mode in temporal_feature_modes:
        normalized = str(mode).strip().lower().replace("-", "_").replace("+", "_")
        values.append(normalized in {"mean_d1", "d1", "mean_delta", "mean_deltas"})
    return sorted(set(values))


def build_neureptrace_supervised_lowrank_config(
    *,
    data_folder: str | Path,
    participants: str | Sequence[int | str] = DEFAULT_PARTICIPANTS,
    time_windows: Sequence[tuple[float, float]] = ((0.0, 0.45),),
    time_bin_size: float = 0.01,
    temporal_feature_modes: Sequence[str] = ("mean",),
    baseline_window: tuple[float, float] = (-0.35, -0.05),
    normalizations: Sequence[str] = ("subject_baseline_whiten",),
    projections: Sequence[str] = ("pls",),
    classifiers: Sequence[str] = ("multinomial-logistic",),
    classifier_params: Sequence[float] = (0.03, 0.1, 0.3, 1.0, 3.0),
    components_values: Sequence[int] = (32, 64, 128),
    ensemble_size: int = 1,
    max_iter: int = 3000,
) -> dict[str, Any]:
    """Translate common PyMEGDec low-rank CLI arguments to a NeuRepTrace config."""

    unsupported_projection = sorted(set(str(item).strip().lower() for item in projections) - {"pls", "supervised_pls"})
    if unsupported_projection:
        raise ValueError("NeuRepTrace supervised-lowrank LOSO supports the PLS/supervised_pls projection only; unsupported: " + ", ".join(unsupported_projection))
    unsupported_normalization = sorted(set(str(item).strip().lower() for item in normalizations) - {"subject_baseline_whiten"})
    if unsupported_normalization:
        raise ValueError("The generated NeuRepTrace lowrank config supports subject_baseline_whiten only. Use a native NeuRepTrace config for other preprocessing modes.")

    windows = [_time_window(window) for window in time_windows]
    baseline = _time_window(baseline_window)
    tmin = min([baseline[0], *[window[0] for window in windows]])
    tmax = max([baseline[1], *[window[1] for window in windows]])
    temporal_bins = [max(1, int(round((stop - start) / float(time_bin_size)))) for start, stop in windows]
    temporal_bins = sorted(set(temporal_bins))
    if isinstance(participants, str):
        participant_config: str | list[int | str] = participants
    else:
        participant_config = list(participants)
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
            "fields": {"trial": "trial", "time": "time", "label": "label", "trialinfo": "trialinfo", "sensor_geometry": "grad"},
        },
        "participants": {"ids": participant_config},
        "validation": {"trim_channel_labels_to_data": True, "channel_policy": "exact"},
        "metadata": {"columns": [{"name": "stimulus_class", "index": 0}, {"name": "condition", "index": 1, "optional": True}]},
        "preprocessing": {"tmin": float(tmin), "tmax": float(tmax), "normalization": "subject_baseline_whiten", "baseline_window": [float(baseline[0]), float(baseline[1])]},
        "decoding": {"label_column": "stimulus_class", "group_column": "participant", "classifier": str(classifiers[0]), "max_iter": int(max_iter)},
        "supervised_lowrank_loso": {
            "group_column": "participant",
            "selection_metric": "balanced_accuracy",
            "ensemble_size": int(ensemble_size),
            "ensemble_aggregation": "log_mean",
            "min_probability": 1e-12,
            "candidate_grid": {
                "temporal_bins": temporal_bins,
                "pls_components": [int(value) for value in components_values],
                "decoders": [str(value) for value in classifiers],
                "c_grid": [float(value) for value in classifier_params],
                "include_deltas": _delta_values(temporal_feature_modes),
                "epoch_windows": [{"name": _window_name(index, window), "start": float(window[0]), "stop": float(window[1])} for index, window in enumerate(windows)],
            },
        },
        "outputs": {
            "base_dir": "outputs/neureptrace_supervised_lowrank_loso",
            "supervised_lowrank_loso_summary_csv": "supervised_lowrank_loso_summary.csv",
            "supervised_lowrank_loso_inner_cv_csv": "supervised_lowrank_loso_inner_cv.csv",
            "supervised_lowrank_loso_predictions_csv": "supervised_lowrank_loso_predictions.csv",
            "provenance": True,
            "hash_input_files": True,
        },
    }


def write_neureptrace_supervised_lowrank_config(config: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Run PyMEGDec's full-epoch low-rank command through NeuRepTrace supervised-lowrank LOSO.")
    parser.add_argument("--neureptrace-config", type=Path, help="Run an existing NeuRepTrace supervised_lowrank_loso config.")
    parser.add_argument("--write-neureptrace-config", type=Path, help="Write the generated NeuRepTrace JSON config before running.")
    parser.add_argument("--data-dir", dest="data_folder", default=None)
    parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS)
    parser.add_argument("--outer-participants", default=None, help="Accepted for legacy compatibility but not supported by the NeuRepTrace wrapper.")
    parser.add_argument("--time-windows", default="0.0:0.45")
    parser.add_argument("--time-bin-size", type=float, default=0.01)
    parser.add_argument("--temporal-feature-modes", default="mean")
    parser.add_argument("--baseline-window", default="-0.35:-0.05")
    parser.add_argument("--normalizations", default="subject_baseline_whiten")
    parser.add_argument("--projections", default="pls")
    parser.add_argument("--classifiers", default="multinomial-logistic")
    parser.add_argument("--classifier-params", default="0.03,0.1,0.3,1.0,3.0")
    parser.add_argument("--components-values", default="32,64,128")
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None, help="Use a native NeuRepTrace config for trial caps.")
    parser.add_argument("--trial-selection", default=None, help="Use a native NeuRepTrace config for trial selection.")
    parser.add_argument("--trial-selection-seed", type=int, default=None, help="Use a native NeuRepTrace config for trial selection.")
    parser.add_argument("--label-shuffle-control", action="store_true", help="Not supported by this NeuRepTrace wrapper.")
    parser.add_argument("--label-shuffle-seed", type=int, default=0)
    parser.add_argument("--chance-classes", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--signflip-permutations", type=int, default=10000)
    parser.add_argument("--signflip-seed", type=int, default=0)
    parser.add_argument("--outer-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_outer.csv")
    parser.add_argument("--summary-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_group_summary.csv")
    parser.add_argument("--inner-validation-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_inner_validation.csv")
    parser.add_argument("--selected-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_selected.csv", help="Compatibility metadata CSV derived from the summary.")
    parser.add_argument("--predictions-output", default="outputs/stimulus_cross_subject_full_epoch_lowrank_predictions.csv")
    parser.add_argument("--confusion-output", default=None)
    parser.add_argument("--per-stimulus-output", default=None)
    parser.add_argument("--confusion-pairs-output", default=None)
    return parser


def _write_selected(summary: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [column for column in ("outer_test_subject", "selected_candidates", "inner_selection_metric", "inner_best_score", "inner_selected_mean_score", "ensemble_size") if column in summary.columns]
    summary[columns].to_csv(path, index=False)


def stimulus_cross_subject_full_epoch_lowrank(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    unsupported = []
    if args.outer_participants:
        unsupported.append("--outer-participants")
    if args.max_trials_per_class_per_participant is not None:
        unsupported.append("--max-trials-per-class-per-participant")
    if args.trial_selection is not None:
        unsupported.append("--trial-selection")
    if args.trial_selection_seed is not None:
        unsupported.append("--trial-selection-seed")
    if args.label_shuffle_control:
        unsupported.append("--label-shuffle-control")
    if unsupported:
        parser.error("The NeuRepTrace-backed full-epoch lowrank wrapper cannot translate: " + ", ".join(unsupported) + ". Use a native NeuRepTrace config.")

    if args.neureptrace_config is not None:
        config_path = args.neureptrace_config
    else:
        data_folder = resolve_data_folder(args.data_folder)
        windows = [_time_window(token) for token in _token_list(args.time_windows)]
        config = build_neureptrace_supervised_lowrank_config(
            data_folder=data_folder,
            participants=args.participants,
            time_windows=windows,
            time_bin_size=args.time_bin_size,
            temporal_feature_modes=_token_list(args.temporal_feature_modes),
            baseline_window=_time_window(args.baseline_window),
            normalizations=_token_list(args.normalizations),
            projections=_token_list(args.projections),
            classifiers=_token_list(args.classifiers),
            classifier_params=_float_list(args.classifier_params),
            components_values=_int_list(args.components_values),
            ensemble_size=args.ensemble_size,
            max_iter=args.max_iter,
        )
        if args.write_neureptrace_config is not None:
            config_path = write_neureptrace_supervised_lowrank_config(config, args.write_neureptrace_config)
        else:
            tmp = tempfile.TemporaryDirectory(prefix="pymegdec-neureptrace-lowrank-")
            config_path = write_neureptrace_supervised_lowrank_config(config, Path(tmp.name) / "supervised_lowrank_loso.json")
    summary = run_supervised_lowrank_loso(
        config_path,
        out_path=args.outer_output,
        inner_cv_out_path=args.inner_validation_output,
        predictions_out_path=args.predictions_output,
    )
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_output, index=False)
    _write_selected(summary, args.selected_output)
    print(f"Wrote {len(summary)} NeuRepTrace supervised-lowrank LOSO rows to {args.outer_output}")
    if args.summary_output:
        print(f"Wrote compatibility group summary to {args.summary_output}")
    return 0


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return stimulus_cross_subject_full_epoch_lowrank(argv, prog)


__all__ = [
    "build_neureptrace_supervised_lowrank_config",
    "stimulus_cross_subject_full_epoch_lowrank",
    "write_neureptrace_supervised_lowrank_config",
]
