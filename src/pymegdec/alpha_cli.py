"""Alpha-analysis command handlers for the grouped PyMEGDec CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from pymegdec.alpha_metrics import (
    DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
    DEFAULT_PROJECTION_REFERENCE_PATTERN,
    export_participant_alpha_metrics,
)
from pymegdec.alpha_movement import (
    DEFAULT_MOVEMENT_TIME_WINDOW,
    DEFAULT_SENSOR_PATTERN,
    DEFAULT_TRAJECTORY_STEP_S,
    AlphaMovementConfig,
    export_alpha_movement,
)
from pymegdec.cli import (
    add_alpha_metric_arguments,
    alpha_metric_config_from_args,
    parse_range,
)
from pymegdec.reaction_time_analysis import (
    DEFAULT_ALPHA_RT_METRICS,
    AlphaReactionTimeExportConfig,
    ReactionTimeCsvConfig,
    available_participants,
    export_alpha_reaction_time_analysis,
    parse_participant_spec,
    write_alpha_reaction_time_plots,
)


def _participants(value: str | None, data_dir, cue: bool) -> list[int]:
    if value:
        return parse_participant_spec(value)
    return available_participants(data_dir, cue=cue)


def _build_alpha_metrics_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export exploratory prestimulus alpha metrics to CSV.")
    parser.add_argument("--data-dir", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument("--participant", type=int, required=True, help="Participant id to export.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--cue",
        action="store_true",
        help="Use Part*CueData.mat instead of Part*Data.mat.",
    )
    add_alpha_metric_arguments(parser)
    return parser


def alpha_metrics(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_alpha_metrics_parser(prog=prog)
    args = parser.parse_args(argv)

    config = alpha_metric_config_from_args(args)
    rows = export_participant_alpha_metrics(
        args.data_dir,
        args.participant,
        args.output,
        cue=args.cue,
        config=config,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


def _build_alpha_movement_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Export sensor-level alpha movement trajectories. The trajectory is " "a MEG sensor-array proxy, not source-localized brain movement."),
    )
    parser.add_argument("--data-dir", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument(
        "--participants",
        default=None,
        help="Participant ids such as 1-4,6,8. Defaults to all available MAT files.",
    )
    parser.add_argument(
        "--trajectory-output",
        required=True,
        help="Output CSV for trial/timepoint sensor-level trajectories.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional output CSV averaged by participant, condition, and time.",
    )
    parser.add_argument(
        "--cue",
        action="store_true",
        help="Use Part*CueData.mat instead of Part*Data.mat.",
    )
    parser.add_argument(
        "--location-pattern",
        default=DEFAULT_SENSOR_PATTERN,
        help="Regex for selecting channels by label. Defaults to all MEG channels.",
    )
    parser.add_argument(
        "--projection-reference-pattern",
        default=DEFAULT_PROJECTION_REFERENCE_PATTERN,
        help=(
            "Regex for channels used to fit the common 2D projection frame. "
            "Defaults to all MEG channels."
        ),
    )
    parser.add_argument(
        "--min-reference-axis-projection",
        type=float,
        default=DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
        help="Minimum robust in-plane projection of a global coordinate axis.",
    )
    parser.add_argument(
        "--time-window",
        type=parse_range,
        default=DEFAULT_MOVEMENT_TIME_WINDOW,
        help="Time window as start,stop in seconds.",
    )
    parser.add_argument(
        "--frequency-range",
        type=parse_range,
        default=(8.0, 12.0),
        help="Frequency range as low,high in Hz.",
    )
    parser.add_argument(
        "--trajectory-step-s",
        type=float,
        default=DEFAULT_TRAJECTORY_STEP_S,
        help="Trajectory sampling step in seconds.",
    )
    return parser


def alpha_movement(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_alpha_movement_parser(prog=prog)
    args = parser.parse_args(argv)

    participants = _participants(args.participants, args.data_dir, args.cue)
    if not participants:
        parser.error("No participants found. Pass --participants or configure a data directory with matching MAT files.")

    config = AlphaMovementConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
        trajectory_step_s=args.trajectory_step_s,
        projection_reference_pattern=args.projection_reference_pattern,
        min_reference_axis_projection=args.min_reference_axis_projection,
    )
    rows, summary_rows = export_alpha_movement(
        args.data_dir,
        participants,
        args.trajectory_output,
        summary_output_path=args.summary_output,
        cue=args.cue,
        config=config,
    )
    print(f"Wrote {len(rows)} trajectory rows to {args.trajectory_output}")
    if args.summary_output:
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    return 0


def _build_alpha_reaction_time_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Analyze prestimulus alpha metrics against reaction time.")
    parser.add_argument("--data-dir", default=None, help="Directory containing Part*Data.mat files.")
    parser.add_argument(
        "--participants",
        default=None,
        help="Participant ids such as 1-4,6,8. Defaults to all available MAT files.",
    )
    parser.add_argument(
        "--reaction-times",
        default=None,
        help="CSV containing participant, trial, and reaction_time columns.",
    )
    parser.add_argument("--alpha-metrics", default=None, help="Optional precomputed alpha metrics CSV.")
    parser.add_argument(
        "--joined-output",
        required=True,
        help="Output CSV for matched trial-level alpha/RT rows.",
    )
    parser.add_argument("--summary-output", required=True, help="Output CSV for association summaries.")
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional directory for simple alpha/RT scatter plots.",
    )
    parser.add_argument(
        "--cue",
        action="store_true",
        help="Use Part*CueData.mat instead of Part*Data.mat.",
    )
    parser.add_argument(
        "--trialinfo-rt-column",
        type=int,
        default=None,
        help="Zero-based trialinfo column containing RT when no CSV is supplied.",
    )
    parser.add_argument(
        "--reaction-time-trial-base",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Index base used by external reaction-time CSV trial numbers. "
            "Use 1 for behavioral files numbered 1..N."
        ),
    )
    parser.add_argument(
        "--reaction-time-scale",
        type=float,
        default=1.0,
        help="Scale applied to RT values, for example 0.001 for milliseconds.",
    )
    parser.add_argument("--participant-column", default=None, help="Reaction-time CSV participant column override.")
    parser.add_argument("--trial-column", default=None, help="Reaction-time CSV trial column override.")
    parser.add_argument("--reaction-time-column", default=None, help="Reaction-time CSV RT column override.")
    parser.add_argument("--dataset-column", default=None, help="Reaction-time CSV dataset column override.")
    add_alpha_metric_arguments(parser)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_ALPHA_RT_METRICS,
        help="Alpha metrics to summarize.",
    )
    return parser


def alpha_reaction_time(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_alpha_reaction_time_parser(prog=prog)
    args = parser.parse_args(argv)

    participants = _participants(args.participants, args.data_dir, args.cue)
    if not participants and (not args.alpha_metrics or not args.reaction_times):
        parser.error("No participants found. Pass --participants or configure a data directory with matching MAT files.")

    default_participant = participants[0] if len(participants) == 1 else None
    alpha_config = alpha_metric_config_from_args(args)
    csv_config = ReactionTimeCsvConfig(
        participant_column=args.participant_column,
        trial_column=args.trial_column,
        reaction_time_column=args.reaction_time_column,
        dataset_column=args.dataset_column,
        default_participant=default_participant,
        default_dataset="cue" if args.cue else "main",
        reaction_time_scale=args.reaction_time_scale,
        trial_index_base=args.reaction_time_trial_base,
    )
    export_config = AlphaReactionTimeExportConfig(
        reaction_times_path=args.reaction_times,
        alpha_metrics_path=args.alpha_metrics,
        joined_output_path=args.joined_output,
        summary_output_path=args.summary_output,
        cue=args.cue,
        alpha_config=alpha_config,
        csv_config=csv_config,
        trialinfo_rt_column=args.trialinfo_rt_column,
        metrics=tuple(args.metrics),
    )

    joined_rows, summary_rows = export_alpha_reaction_time_analysis(
        args.data_dir,
        participants,
        config=export_config,
    )
    if args.plots_dir:
        write_alpha_reaction_time_plots(joined_rows, args.plots_dir, metrics=args.metrics)
    print(f"Wrote {len(joined_rows)} matched trial rows to {args.joined_output}")
    print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    return 0
