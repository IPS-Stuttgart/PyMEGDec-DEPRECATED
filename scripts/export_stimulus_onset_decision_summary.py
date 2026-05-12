"""Build a decision-oriented summary from stimulus onset benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_PRIMARY_SETTINGS = (
    "max_run_sweep_selected",
    "point_sweep_selected",
    "robust_max_run",
    "point_baseline",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="outputs")
    parser.add_argument("--quality-summary", default=None)
    parser.add_argument("--chance-rate", type=float, default=0.0625)
    parser.add_argument("--max-false-alarm-rate", type=float, default=0.05)
    parser.add_argument(
        "--primary-settings",
        default=",".join(DEFAULT_PRIMARY_SETTINGS),
        help="Comma-separated settings to report in the primary comparison block.",
    )
    parser.add_argument("--output-csv", default="outputs/stimulus_onset_decision_summary.csv")
    parser.add_argument("--output-markdown", default="outputs/stimulus_onset_decision_summary.md")
    parser.add_argument("--output-json", default="outputs/stimulus_onset_decision_summary.json")
    return parser


def _render_markdown(
    table: pd.DataFrame,
    *,
    primary_settings: tuple[str, ...],
    accepted_setting: str | None,
    chance_rate: float,
    max_false_alarm_rate: float,
) -> str:
    primary = table.loc[table["setting"].isin(primary_settings)].copy()
    if primary.empty:
        primary = table.copy()
    column_order = [
        "setting",
        "false_alarm_rate_mean",
        "post_stimulus_detected_rate_mean",
        "correct_detection_rate_mean",
        "chance_floor",
        "above_chance_margin",
        "accepted",
        "median_latency_s",
    ]
    primary = primary[column_order]
    table_block = "```\n" + primary.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n```"
    lines = [
        "# Stimulus onset decision summary",
        "",
        f"Acceptance rule: `false_alarm_rate_mean <= {max_false_alarm_rate}` and " f"`correct_detection_rate_mean > {chance_rate} * post_stimulus_detected_rate_mean`.",
        "",
        table_block,
        "",
    ]
    if accepted_setting:
        lines.append(f"Recommended setting: `{accepted_setting}`.")
    else:
        lines.append("Recommended setting: none passed the acceptance rule.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact_dir = Path(args.artifact_dir)
    quality_summary = Path(args.quality_summary) if args.quality_summary else artifact_dir / "stimulus_onset_quality_summary.csv"
    table = pd.read_csv(quality_summary).copy()
    for column in (
        "false_alarm_rate_mean",
        "post_stimulus_detected_rate_mean",
        "correct_detection_rate_mean",
        "median_latency_s",
        "mean_latency_s",
    ):
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    table["chance_floor"] = args.chance_rate * table["post_stimulus_detected_rate_mean"]
    table["above_chance_margin"] = table["correct_detection_rate_mean"] - table["chance_floor"]
    table["passes_false_alarm"] = table["false_alarm_rate_mean"] <= args.max_false_alarm_rate
    table["passes_chance"] = table["above_chance_margin"] > 0.0
    table["accepted"] = table["passes_false_alarm"] & table["passes_chance"]
    accepted = table.loc[table["accepted"]].copy()
    accepted_setting = None
    if not accepted.empty:
        accepted = accepted.sort_values(
            by=["post_stimulus_detected_rate_mean", "correct_detection_rate_mean", "median_latency_s", "false_alarm_rate_mean"],
            ascending=[False, False, True, True],
        )
        accepted_setting = str(accepted.iloc[0]["setting"])
    primary_settings = tuple(item.strip() for item in args.primary_settings.split(",") if item.strip())
    markdown = _render_markdown(
        table,
        primary_settings=primary_settings,
        accepted_setting=accepted_setting,
        chance_rate=args.chance_rate,
        max_false_alarm_rate=args.max_false_alarm_rate,
    )
    Path(args.output_csv).write_text(table.to_csv(index=False), encoding="utf-8")
    Path(args.output_markdown).write_text(markdown, encoding="utf-8")
    payload = {
        "accepted_setting": accepted_setting,
        "chance_rate": args.chance_rate,
        "max_false_alarm_rate": args.max_false_alarm_rate,
        "rows": json.loads(table.to_json(orient="records")),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
