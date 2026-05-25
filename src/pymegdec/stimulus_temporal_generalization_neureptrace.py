"""NeuRepTrace-backed temporal-generalization compatibility workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from neureptrace.decoding.temporal_generalization import summarize_temporal_generalization_matrix

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.cli import normalize_argv
from pymegdec.data_config import resolve_data_folder
from pymegdec.stimulus_decoding import (
    TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS,
    StimulusDecodingConfig,
    evaluate_participant_stimulus_temporal_generalization,
    window_centers_from_range,
)


def _present_group_fields(rows: Sequence[dict[str, Any]], preferred_fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(field for field in preferred_fields if any(field in row for row in rows))


def summarize_stimulus_temporal_generalization_with_neureptrace(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize PyMEGDec temporal-generalization rows via NeuRepTrace."""

    if not rows:
        return []
    frame = pd.DataFrame(list(rows))
    group_fields = _present_group_fields(list(rows), TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    summary = summarize_temporal_generalization_matrix(
        frame,
        group_columns=group_fields,
        accuracy_column="accuracy",
        chance_column="chance_accuracy",
    )
    if "n_rows" in summary.columns and "n_participants" not in summary.columns:
        summary.insert(len(group_fields), "n_participants", summary["n_rows"].astype(int))
    if "n_rows" in summary.columns:
        summary = summary.drop(columns=["n_rows"])
    return summary.to_dict(orient="records")


def export_stimulus_temporal_generalization(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    config=None,
    progress=None,
):
    """Run BUSH-MEG temporal generalization and summarize with NeuRepTrace."""

    config = config or StimulusDecodingConfig()
    data_folder = resolve_data_folder(data_folder)
    rows: list[dict[str, Any]] = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(evaluate_participant_stimulus_temporal_generalization(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_temporal_generalization_with_neureptrace(rows)
    if summary_output_path:
        write_alpha_metrics_csv(summary_rows, summary_output_path)
    return rows, summary_rows


def stimulus_temporal_generalization(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    from pymegdec import stimulus_cli as _legacy_stimulus_cli

    parser = _legacy_stimulus_cli._build_temporal_generalization_parser(prog=prog)  # pylint: disable=protected-access
    args = parser.parse_args(normalize_argv(argv))
    data_folder = resolve_data_folder(args.data_folder)
    participants = _legacy_stimulus_cli._participants_or_error(parser, args.participants, data_folder)  # pylint: disable=protected-access
    config = _legacy_stimulus_cli._base_config(  # pylint: disable=protected-access
        args,
        window_centers=window_centers_from_range(args.time_window, args.window_step_s),
    )
    rows, summary_rows = export_stimulus_temporal_generalization(
        data_folder,
        participants,
        args.output,
        summary_output_path=args.summary_output,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Wrote {len(rows)} participant/train/test rows to {args.output}")
    print(f"Wrote {len(summary_rows)} train/test summary rows to {args.summary_output}")
    return 0


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    return stimulus_temporal_generalization(argv, prog)


__all__ = [
    "export_stimulus_temporal_generalization",
    "stimulus_temporal_generalization",
    "summarize_stimulus_temporal_generalization_with_neureptrace",
]
