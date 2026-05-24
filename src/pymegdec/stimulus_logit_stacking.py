"""Compatibility bridge for source-OOF logit/probability stacking.

The reusable stacker now lives in :mod:`neureptrace.probability_stacking`.
PyMEGDec keeps the historical ``pymegdec stimulus cross-subject-logit-stack``
entry point as a compatibility wrapper for already-exported probability
observation tables.

This deliberately stops training BUSH-MEG base models inside PyMEGDec. Generate
source-OOF and target probability tables with NeuRepTrace workflows, then stack
those tables here or, preferably, with::

    neureptrace source-oof-stacking --source-oof source.csv --target target.csv --out stacked.csv
"""

from __future__ import annotations

import argparse
import glob
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from neureptrace import probability_stacking as _nrt_stacking
from neureptrace.probability_stacking import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_ITER,
    DEFAULT_MIN_PROBABILITY,
    DEFAULT_OUTPUT_EMISSION_MODE,
    DEFAULT_TEMPERATURE,
    SourceOOFStackingFit,
    align_probability_cube,
    combine_probability_cube,
    fit_source_oof_stacking,
    stack_probability_observations,
    summarize_stacked_metrics,
)

from pymegdec.cli import normalize_argv

LOGIT_STACK_CLASSIFIER = "source_oof_logit_stack"
DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION = "none"
LOGIT_STACK_SCORE_NORMALIZATION_MODES = ("none", "row_z", "rank")
DEFAULT_LOGIT_STACK_WEIGHTING = "stacked"
LOGIT_STACK_WEIGHTING_MODES = ("uniform", "softmax", "stacked", "inner_softmax", "greedy_balanced")
DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE = DEFAULT_TEMPERATURE
DEFAULT_LOGIT_STACK_CLASS_BIAS = False
DEFAULT_LOGIT_STACK_CLASS_BIAS_L2 = 0.0
DEFAULT_LOGIT_STACK_MAX_BASE_MODELS = None

# Keep the historical public-ish name, but point it at the NeuRepTrace fit type.
LogitStackingFit = SourceOOFStackingFit


def _normalize_weighting(value: str) -> str:
    """Normalize historical PyMEGDec weighting names to NeuRepTrace names."""

    token = str(value).strip().lower().replace("-", "_")
    aliases = {
        "inner_softmax": "softmax",
        "greedy_balanced": "stacked",
    }
    token = aliases.get(token, token)
    if token not in _nrt_stacking.WEIGHTING_MODES:
        raise ValueError(f"weighting must be one of {sorted(_nrt_stacking.WEIGHTING_MODES)} or PyMEGDec aliases inner_softmax/greedy_balanced.")
    return token


def _normalize_score_normalization(value: str) -> str:
    """Validate the old score-normalization option.

    NeuRepTrace stacks calibrated probability rows, so this option is retained
    only so old scripts receive a clear warning instead of an import error.
    """

    token = str(value).strip().lower().replace("-", "_")
    if token not in LOGIT_STACK_SCORE_NORMALIZATION_MODES:
        raise ValueError(f"score_normalization must be one of {LOGIT_STACK_SCORE_NORMALIZATION_MODES}.")
    return token


def _normalize_max_base_models(value: Any) -> int | None:
    """Parse the deprecated base-model cap option."""

    if value in (None, "", "none", "all"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("max_base_models must be positive, all, or none.")
    return parsed


def _legacy_raw_workflow_error() -> RuntimeError:
    return RuntimeError(
        "PyMEGDec no longer owns source-only logit-stack training from Part*Data.mat files. "
        "Generate NeuRepTrace probability observation tables first and then use "
        "neureptrace source-oof-stacking, or call pymegdec stimulus cross-subject-logit-stack "
        "with --source-oof and --target."
    )


def make_logit_stack_candidate_configs(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
    """Deprecated placeholder for the removed PyMEGDec-native base-model grid."""

    raise _legacy_raw_workflow_error()


def evaluate_cross_subject_logit_stacking(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Deprecated placeholder for the removed PyMEGDec-native raw-data workflow."""

    raise _legacy_raw_workflow_error()


def export_cross_subject_logit_stacking(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Deprecated placeholder for the removed PyMEGDec-native raw-data workflow."""

    raise _legacy_raw_workflow_error()


def _read_csv_inputs(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in paths:
        matches = sorted(glob.glob(str(pattern)))
        if not matches and Path(pattern).exists():
            matches = [str(pattern)]
        if not matches:
            raise FileNotFoundError(f"No CSV files match {pattern!r}.")
        frames.extend(pd.read_csv(path) for path in matches)
    if not frames:
        raise ValueError("No CSV inputs were provided.")
    return pd.concat(frames, ignore_index=True)


def _write_frame(frame: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _selected_row_from_stacked(stacked: pd.DataFrame) -> pd.DataFrame:
    if stacked.empty:
        return pd.DataFrame()
    first = stacked.iloc[0]
    row = {
        "outer_evaluation_mode": LOGIT_STACK_CLASSIFIER,
        "backend": first.get("backend", "source_oof_stacking"),
        "decoder": first.get("decoder", LOGIT_STACK_CLASSIFIER),
        "source_oof_candidates": first.get("source_oof_candidates", ""),
        "source_oof_weights": first.get("source_oof_weights", ""),
        "source_oof_weighting": first.get("source_oof_weighting", ""),
        "source_oof_temperature": first.get("source_oof_temperature", ""),
        "source_oof_balanced_accuracy": first.get("source_oof_balanced_accuracy", ""),
        "source_oof_log_loss": first.get("source_oof_log_loss", ""),
        "source_oof_alignment_columns": first.get("source_oof_alignment_columns", ""),
        "model_hash": first.get("model_hash", ""),
    }
    return pd.DataFrame([row])


def _write_prediction_derivatives(
    stacked: pd.DataFrame,
    *,
    confusion_output: str | Path | None = None,
    per_stimulus_output: str | Path | None = None,
    confusion_pairs_output: str | Path | None = None,
) -> None:
    if not {"true_label", "predicted_label"}.issubset(stacked.columns):
        return

    if confusion_output is not None:
        confusion = stacked.groupby(["true_label", "predicted_label"], dropna=False).size().reset_index(name="count")
        _write_frame(confusion, confusion_output)

    if per_stimulus_output is not None:
        per = stacked.assign(is_correct=stacked["true_label"].astype(str) == stacked["predicted_label"].astype(str))
        per_stimulus = (
            per.groupby("true_label", dropna=False)
            .agg(n_trials=("true_label", "size"), n_correct=("is_correct", "sum"), accuracy=("is_correct", "mean"))
            .reset_index()
        )
        _write_frame(per_stimulus, per_stimulus_output)

    if confusion_pairs_output is not None:
        numeric = stacked.copy()
        numeric["true_label_numeric"] = pd.to_numeric(numeric["true_label"], errors="coerce")
        numeric["predicted_label_numeric"] = pd.to_numeric(numeric["predicted_label"], errors="coerce")
        errors = numeric.loc[
            numeric["true_label_numeric"].notna()
            & numeric["predicted_label_numeric"].notna()
            & (numeric["true_label_numeric"] != numeric["predicted_label_numeric"]),
            ["true_label_numeric", "predicted_label_numeric"],
        ].copy()
        if errors.empty:
            pairs = pd.DataFrame(columns=["stimulus_a", "stimulus_b", "count"])
        else:
            errors["stimulus_a"] = errors[["true_label_numeric", "predicted_label_numeric"]].min(axis=1).astype(int)
            errors["stimulus_b"] = errors[["true_label_numeric", "predicted_label_numeric"]].max(axis=1).astype(int)
            pairs = errors.groupby(["stimulus_a", "stimulus_b"], dropna=False).size().reset_index(name="count")
        _write_frame(pairs, confusion_pairs_output)


def stack_source_oof_observations(
    source_oof_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    candidate_column: str = _nrt_stacking.DEFAULT_CANDIDATE_COLUMN,
    candidates: Sequence[str] | None = None,
    alignment_columns: Sequence[str] | None = None,
    weighting: str = DEFAULT_LOGIT_STACK_WEIGHTING,
    temperature: float | None = DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    output_decoder: str = LOGIT_STACK_CLASSIFIER,
    output_emission_mode: str = DEFAULT_OUTPUT_EMISSION_MODE,
) -> pd.DataFrame:
    """Stack source-OOF probability observations using NeuRepTrace."""

    return stack_probability_observations(
        source_oof_observations,
        target_observations,
        candidate_column=candidate_column,
        candidates=candidates,
        alignment_columns=alignment_columns,
        weighting=_normalize_weighting(weighting),
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_probability=min_probability,
        output_decoder=output_decoder,
        output_emission_mode=output_emission_mode,
    )


def run_source_oof_probability_stacking(
    *,
    source_oof_paths: Sequence[str | Path],
    target_paths: Sequence[str | Path],
    predictions_output_path: str | Path,
    metrics_output_path: str | Path | None = None,
    group_summary_output_path: str | Path | None = None,
    selected_output_path: str | Path | None = None,
    confusion_output_path: str | Path | None = None,
    per_stimulus_output_path: str | Path | None = None,
    confusion_pairs_output_path: str | Path | None = None,
    candidate_column: str = _nrt_stacking.DEFAULT_CANDIDATE_COLUMN,
    candidates: Sequence[str] | None = None,
    alignment_columns: Sequence[str] | None = None,
    weighting: str = DEFAULT_LOGIT_STACK_WEIGHTING,
    temperature: float | None = DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    output_decoder: str = LOGIT_STACK_CLASSIFIER,
    output_emission_mode: str = DEFAULT_OUTPUT_EMISSION_MODE,
) -> dict[str, pd.DataFrame]:
    """Read source/target observation CSVs, delegate stacking to NeuRepTrace, and write compatibility outputs."""

    source_oof = _read_csv_inputs(source_oof_paths)
    target = _read_csv_inputs(target_paths)
    stacked = stack_source_oof_observations(
        source_oof,
        target,
        candidate_column=candidate_column,
        candidates=candidates,
        alignment_columns=alignment_columns,
        weighting=weighting,
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_probability=min_probability,
        output_decoder=output_decoder,
        output_emission_mode=output_emission_mode,
    )
    metrics = summarize_stacked_metrics(stacked)
    selected = _selected_row_from_stacked(stacked)

    _write_frame(stacked, predictions_output_path)
    _write_frame(metrics, metrics_output_path)
    _write_frame(metrics, group_summary_output_path)
    _write_frame(selected, selected_output_path)
    _write_prediction_derivatives(
        stacked,
        confusion_output=confusion_output_path,
        per_stimulus_output=per_stimulus_output_path,
        confusion_pairs_output=confusion_pairs_output_path,
    )
    return {"predictions": stacked, "metrics": metrics, "selected": selected}


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Compatibility wrapper for NeuRepTrace source-OOF probability stacking.",
    )

    parser.add_argument("--source-oof", nargs="+", help="Source out-of-fold probability observation CSVs/globs used to fit weights.")
    parser.add_argument("--target", nargs="+", help="Target probability observation CSVs/globs to ensemble with source-fitted weights.")
    parser.add_argument("--out", dest="predictions_output", type=Path, help="Stacked target observation CSV. Alias for --predictions-output.")
    parser.add_argument("--predictions-output", dest="predictions_output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_predictions.csv"))
    parser.add_argument("--metrics-output", type=Path, help="Optional NeuRepTrace grouped metrics CSV.")

    parser.add_argument("--candidate-column", default=_nrt_stacking.DEFAULT_CANDIDATE_COLUMN)
    parser.add_argument("--candidate", action="append", dest="candidates", help="Candidate/decoder to include. May be repeated; defaults to source order.")
    parser.add_argument("--alignment-column", action="append", dest="alignment_columns", help="Alignment key column. May be repeated; defaults to NeuRepTrace observation keys.")
    parser.add_argument("--stacker-weighting", choices=sorted(LOGIT_STACK_WEIGHTING_MODES), default=DEFAULT_LOGIT_STACK_WEIGHTING)
    parser.add_argument("--stacker-weighting-temperature", type=float, default=DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-probability", type=float, default=DEFAULT_MIN_PROBABILITY)
    parser.add_argument("--output-decoder", default=LOGIT_STACK_CLASSIFIER)
    parser.add_argument("--output-emission-mode", default=DEFAULT_OUTPUT_EMISSION_MODE)

    # Legacy output names retained as compatibility derivatives.
    parser.add_argument("--outer-output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_outer.csv"), help="Compatibility metrics CSV.")
    parser.add_argument("--summary-output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_group_summary.csv"), help="Compatibility metrics summary CSV.")
    parser.add_argument("--selected-output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_selected.csv"), help="Stacking weights/metadata CSV.")
    parser.add_argument("--inner-validation-output", type=Path, help="Ignored compatibility option; source OOF rows are supplied through --source-oof.")
    parser.add_argument("--confusion-output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_confusion.csv"))
    parser.add_argument("--per-stimulus-output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_per_stimulus.csv"))
    parser.add_argument("--confusion-pairs-output", type=Path, default=Path("outputs/stimulus_cross_subject_logit_stack_confusion_pairs.csv"))

    # Raw-data legacy options accepted so old command lines fail with one clear
    # migration message instead of many "unknown argument" errors.
    parser.add_argument("--data-dir", dest="data_folder", help=argparse.SUPPRESS)
    parser.add_argument("--participants", help=argparse.SUPPRESS)
    parser.add_argument("--outer-participants", help=argparse.SUPPRESS)
    parser.add_argument("--window-centers", help=argparse.SUPPRESS)
    parser.add_argument("--window-size", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-window", help=argparse.SUPPRESS)
    parser.add_argument("--feature-modes", help=argparse.SUPPRESS)
    parser.add_argument("--normalizations", help=argparse.SUPPRESS)
    parser.add_argument("--alignments", help=argparse.SUPPRESS)
    parser.add_argument("--classifiers", help=argparse.SUPPRESS)
    parser.add_argument("--classifier-params", help=argparse.SUPPRESS)
    parser.add_argument("--components-pca-values", help=argparse.SUPPRESS)
    parser.add_argument("--sample-weightings", help=argparse.SUPPRESS)
    parser.add_argument("--score-calibrations", help=argparse.SUPPRESS)
    parser.add_argument("--alignment-alphas", help=argparse.SUPPRESS)
    parser.add_argument("--max-trials-per-class-per-participant", help=argparse.SUPPRESS)
    parser.add_argument("--trial-selection", help=argparse.SUPPRESS)
    parser.add_argument("--trial-selection-seed", help=argparse.SUPPRESS)
    parser.add_argument("--stacker-score-normalization", default=DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION, help=argparse.SUPPRESS)
    parser.add_argument("--stacker-max-base-models", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-stacker-class-bias", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stacker-class-bias-l2", type=float, default=DEFAULT_LOGIT_STACK_CLASS_BIAS_L2, help=argparse.SUPPRESS)
    parser.add_argument("--chance-classes", help=argparse.SUPPRESS)
    parser.add_argument("--random-state", help=argparse.SUPPRESS)
    parser.add_argument("--label-shuffle-control", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label-shuffle-seed", help=argparse.SUPPRESS)
    parser.add_argument("--signflip-permutations", help=argparse.SUPPRESS)
    parser.add_argument("--signflip-seed", help=argparse.SUPPRESS)
    return parser


def _warn_legacy_noops(args: argparse.Namespace) -> None:
    score_normalization = _normalize_score_normalization(args.stacker_score_normalization)
    if score_normalization != "none":
        warnings.warn(
            "stacker-score-normalization is ignored by the NeuRepTrace probability stacker because inputs are already probabilities.",
            RuntimeWarning,
            stacklevel=2,
        )
    if _normalize_max_base_models(args.stacker_max_base_models) is not None:
        warnings.warn(
            "stacker-max-base-models is ignored by the NeuRepTrace probability stacker; select candidates explicitly with --candidate.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.no_stacker_class_bias or args.stacker_class_bias_l2 != DEFAULT_LOGIT_STACK_CLASS_BIAS_L2:
        warnings.warn(
            "PyMEGDec class-bias calibration is not part of the NeuRepTrace probability stacker and is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
    if args.label_shuffle_control:
        warnings.warn(
            "label-shuffle-control is ignored here; shuffle labels before producing the source/target observation tables if needed.",
            RuntimeWarning,
            stacklevel=2,
        )


def stimulus_cross_subject_logit_stack(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(normalize_argv(argv))
    if not args.source_oof or not args.target:
        parser.error(
            "PyMEGDec no longer trains source-OOF logit stacking from raw Part*Data.mat files. "
            "Pass --source-oof and --target probability observation CSVs, or use "
            "`neureptrace source-oof-stacking` directly."
        )

    try:
        _warn_legacy_noops(args)
        artifacts = run_source_oof_probability_stacking(
            source_oof_paths=args.source_oof,
            target_paths=args.target,
            predictions_output_path=args.predictions_output,
            metrics_output_path=args.metrics_output,
            group_summary_output_path=args.summary_output,
            selected_output_path=args.selected_output,
            confusion_output_path=args.confusion_output,
            per_stimulus_output_path=args.per_stimulus_output,
            confusion_pairs_output_path=args.confusion_pairs_output,
            candidate_column=args.candidate_column,
            candidates=args.candidates,
            alignment_columns=args.alignment_columns,
            weighting=args.stacker_weighting,
            temperature=args.stacker_weighting_temperature,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            min_probability=args.min_probability,
            output_decoder=args.output_decoder,
            output_emission_mode=args.output_emission_mode,
        )
        # Preserve the historical "outer-output" slot by writing the same metric
        # table used for the compatibility summary.
        _write_frame(artifacts["metrics"], args.outer_output)
    except Exception as exc:  # pragma: no cover - exercised by CLI usage
        parser.error(str(exc))

    print(f"Wrote {len(artifacts['predictions'])} stacked probability rows to {args.predictions_output}")
    print(f"Wrote {len(artifacts['metrics'])} stacked metric rows to {args.summary_output}")
    print(f"Wrote {len(artifacts['selected'])} stacking metadata rows to {args.selected_output}")
    return 0


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return stimulus_cross_subject_logit_stack(argv, prog)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LOGIT_STACK_CLASS_BIAS",
    "DEFAULT_LOGIT_STACK_CLASS_BIAS_L2",
    "DEFAULT_LOGIT_STACK_MAX_BASE_MODELS",
    "DEFAULT_LOGIT_STACK_SCORE_NORMALIZATION",
    "DEFAULT_LOGIT_STACK_WEIGHTING",
    "DEFAULT_LOGIT_STACK_WEIGHTING_TEMPERATURE",
    "LOGIT_STACK_CLASSIFIER",
    "LOGIT_STACK_SCORE_NORMALIZATION_MODES",
    "LOGIT_STACK_WEIGHTING_MODES",
    "LogitStackingFit",
    "align_probability_cube",
    "combine_probability_cube",
    "evaluate_cross_subject_logit_stacking",
    "export_cross_subject_logit_stacking",
    "fit_source_oof_stacking",
    "make_logit_stack_candidate_configs",
    "run_source_oof_probability_stacking",
    "stack_probability_observations",
    "stack_source_oof_observations",
    "stimulus_cross_subject_logit_stack",
    "summarize_stacked_metrics",
]
