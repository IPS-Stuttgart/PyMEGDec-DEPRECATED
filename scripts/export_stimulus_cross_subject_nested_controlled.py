"""Export nested cross-subject benchmark with optional training-label controls."""

from __future__ import annotations

import argparse
import ast
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pymegdec import stimulus_cross_subject as base  # noqa: E402
from pymegdec.stimulus_cross_subject import (  # noqa: E402
    AUTO_CLASSIFIER_PARAM_GRID_TOKEN,
    AUTO_COMPONENTS_PCA_GRID_TOKEN,
    make_cross_subject_candidate_configs,
)
from pymegdec.stimulus_cross_subject_controls import (  # noqa: E402
    LABEL_CONTROL_MODES,
    evaluate_nested_cross_subject_stimulus_controlled,
    normalize_label_control,
)


def _parse_participants(value: str) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return tuple()
    participants: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, stop_text = token.split("-", maxsplit=1)
            participants.extend(range(int(start_text), int(stop_text) + 1))
        else:
            participants.append(int(token))
    return tuple(participants)


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one float value is required.")
    return values


def _parse_token_list(value: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_time_window(value: str) -> tuple[float, float]:
    start, stop = _parse_float_list(value)
    if start >= stop:
        raise argparse.ArgumentTypeError("Window start must be before stop.")
    return start, stop


def _parse_int_or_inf(value: str):
    value = str(value).strip().lower()
    if value in {"inf", "infinity"}:
        return float("inf")
    return int(value)


def _parse_int_or_inf_list(value: str) -> tuple[int | float | str, ...]:
    values = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower().replace("_", "-") == AUTO_COMPONENTS_PCA_GRID_TOKEN:
            values.append(AUTO_COMPONENTS_PCA_GRID_TOKEN)
        else:
            values.append(_parse_int_or_inf(token))
    if not values:
        raise argparse.ArgumentTypeError("At least one PCA value is required.")
    return tuple(values)


def _parse_classifier_params(value: str) -> tuple[object, ...]:
    params = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        normalized = token.lower().replace("_", "-")
        if normalized in {"default", "nan"}:
            params.append(float("nan"))
        elif normalized == AUTO_CLASSIFIER_PARAM_GRID_TOKEN:
            params.append(AUTO_CLASSIFIER_PARAM_GRID_TOKEN)
        else:
            try:
                parsed = ast.literal_eval(token)
            except (SyntaxError, ValueError):
                parsed = token
            if isinstance(parsed, (int, float)) and math.isnan(float(parsed)):
                parsed = float("nan")
            params.append(parsed)
    if not params:
        raise argparse.ArgumentTypeError("At least one classifier parameter value is required.")
    return tuple(params)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", dest="data_folder", required=True)
    parser.add_argument("--participants", type=_parse_participants, required=True)
    parser.add_argument("--outer-participants", type=_parse_participants, default=tuple())
    parser.add_argument("--window-centers", type=_parse_float_list, required=True)
    parser.add_argument("--window-size", type=float, required=True)
    parser.add_argument("--baseline-window", type=_parse_time_window, required=True)
    parser.add_argument("--feature-modes", type=_parse_token_list, required=True)
    parser.add_argument("--normalizations", type=_parse_token_list, required=True)
    parser.add_argument("--classifiers", type=_parse_token_list, required=True)
    parser.add_argument("--classifier-params", type=_parse_classifier_params, required=True)
    parser.add_argument("--components-pca-values", type=_parse_int_or_inf_list, required=True)
    parser.add_argument("--max-trials-per-class-per-participant", type=int, default=None)
    parser.add_argument("--chance-classes", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--signflip-permutations", type=int, default=10000)
    parser.add_argument("--signflip-seed", type=int, default=0)
    parser.add_argument("--label-control", default="none", choices=LABEL_CONTROL_MODES)
    parser.add_argument("--label-control-seed", type=int, default=0)
    parser.add_argument("--outer-output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--inner-validation-output", default=None)
    parser.add_argument("--selected-output", default=None)
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--confusion-output", default=None)
    parser.add_argument("--per-stimulus-output", default=None)
    parser.add_argument("--confusion-pairs-output", default=None)
    parser.add_argument("--write-incremental", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidate_configs = make_cross_subject_candidate_configs(
        window_centers=args.window_centers,
        window_size=args.window_size,
        baseline_window=args.baseline_window,
        feature_modes=args.feature_modes,
        normalizations=args.normalizations,
        classifiers=args.classifiers,
        classifier_params=args.classifier_params,
        components_pca_values=args.components_pca_values,
        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,
        chance_classes=args.chance_classes,
        random_state=args.random_state,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    label_control = normalize_label_control(args.label_control)
    output_kwargs = {
        "outer_output_path": args.outer_output,
        "group_summary_output_path": args.summary_output,
        "inner_validation_output_path": args.inner_validation_output,
        "selected_output_path": args.selected_output,
        "predictions_output_path": args.predictions_output,
        "confusion_output_path": args.confusion_output,
        "per_stimulus_output_path": args.per_stimulus_output,
        "confusion_pairs_output_path": args.confusion_pairs_output,
    }
    existing = None
    if args.write_incremental:
        existing = base._read_nested_output_rows(  # pylint: disable=protected-access
            outer_output_path=args.outer_output,
            inner_validation_output_path=args.inner_validation_output,
            selected_output_path=args.selected_output,
            predictions_output_path=args.predictions_output,
        )

    def _write_incremental(artifacts):
        if args.write_incremental:
            base._write_nested_output_rows(artifacts, **output_kwargs)  # pylint: disable=protected-access

    artifacts = evaluate_nested_cross_subject_stimulus_controlled(
        args.data_folder,
        args.participants,
        candidate_configs=candidate_configs,
        outer_participants=args.outer_participants or None,
        label_control=label_control,
        label_control_seed=args.label_control_seed,
        existing_artifacts=existing,
        after_outer_fold=_write_incremental,
        progress=lambda message: print(message, flush=True),
    )
    base._write_nested_output_rows(artifacts, **output_kwargs)  # pylint: disable=protected-access
    print(f"Wrote {len(artifacts['outer'])} untouched outer participant rows to {args.outer_output}")
    if args.inner_validation_output:
        print(f"Wrote {len(artifacts['inner_validation'])} inner validation rows to {args.inner_validation_output}")
    if args.selected_output:
        print(f"Wrote {len(artifacts['selected'])} selected hyperparameter rows to {args.selected_output}")
    if args.summary_output:
        print(f"Wrote {len(artifacts['group_summary'])} group summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
