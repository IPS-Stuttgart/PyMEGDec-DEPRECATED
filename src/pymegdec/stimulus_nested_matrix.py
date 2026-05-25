"""Aggregate sharded nested cross-subject stimulus benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.stimulus_cross_subject import (
    summarize_cross_subject_confusion_pairs,
    summarize_cross_subject_predictions,
    summarize_nested_cross_subject_stimulus,
)

SHARD_OUTER_RE = re.compile(r"^matrix_(?P<bundle>.+)_p\d+(?:-p\d+)*_outer\.csv$")
SHARD_SUFFIXES = {
    "outer": "_outer.csv",
    "inner_validation": "_inner_validation.csv",
    "selected": "_selected.csv",
    "predictions": "_predictions.csv",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _with_bundle(rows: Iterable[dict], bundle: str) -> list[dict]:
    return [{**row, "matrix_config_bundle": bundle} for row in rows]


def _write_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_alpha_metrics_csv(_rows_with_consistent_fields(rows), path)


def _rows_with_consistent_fields(rows: list[dict]) -> list[dict]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def _shard_path(outer_path: Path, kind: str) -> Path:
    return outer_path.with_name(outer_path.name.removesuffix(SHARD_SUFFIXES["outer"]) + SHARD_SUFFIXES[kind])


def discover_nested_matrix_shards(input_dir: Path) -> dict[str, list[Path]]:
    """Return outer shard CSVs grouped by matrix configuration bundle."""

    shards: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(input_dir.rglob("matrix_*_p*_outer.csv")):
        match = SHARD_OUTER_RE.match(path.name)
        if match:
            shards[match.group("bundle")].append(path)
    return dict(shards)


def aggregate_nested_matrix_outputs(
    input_dir: Path,
    output_dir: Path,
    *,
    output_stem: str = "nested_matrix",
    signflip_permutations: int = 10_000,
    signflip_seed: int = 0,
) -> dict[str, list[dict]]:
    """Combine completed nested matrix shards and recompute bundle summaries."""

    shards_by_bundle = discover_nested_matrix_shards(input_dir)
    if not shards_by_bundle:
        raise ValueError(f"No nested matrix outer shard CSVs found below {input_dir}.")

    all_outer: list[dict] = []
    all_inner: list[dict] = []
    all_selected: list[dict] = []
    all_predictions: list[dict] = []
    all_summaries: list[dict] = []
    all_confusion: list[dict] = []
    all_per_stimulus: list[dict] = []
    all_confusion_pairs: list[dict] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for bundle, outer_paths in sorted(shards_by_bundle.items()):
        outer_rows: list[dict] = []
        inner_rows: list[dict] = []
        selected_rows: list[dict] = []
        prediction_rows: list[dict] = []
        for outer_path in outer_paths:
            outer_rows.extend(_with_bundle(_read_rows(outer_path), bundle))
            inner_rows.extend(_with_bundle(_read_rows(_shard_path(outer_path, "inner_validation")), bundle))
            selected_rows.extend(_with_bundle(_read_rows(_shard_path(outer_path, "selected")), bundle))
            prediction_rows.extend(_with_bundle(_read_rows(_shard_path(outer_path, "predictions")), bundle))

        summary_rows = summarize_nested_cross_subject_stimulus(
            outer_rows,
            signflip_permutations=signflip_permutations,
            signflip_seed=signflip_seed,
        )
        summary_rows = [{**row, "matrix_config_bundle": bundle} for row in summary_rows]
        confusion_rows, per_stimulus_rows = summarize_cross_subject_predictions(prediction_rows)
        confusion_pair_rows = summarize_cross_subject_confusion_pairs(prediction_rows)
        bundle_confusion_rows = _with_bundle(confusion_rows, bundle)
        bundle_per_stimulus_rows = _with_bundle(per_stimulus_rows, bundle)
        bundle_confusion_pair_rows = _with_bundle(confusion_pair_rows, bundle)

        bundle_stem = f"{output_stem}_{bundle}"
        _write_rows(output_dir / f"{bundle_stem}_outer.csv", outer_rows)
        _write_rows(output_dir / f"{bundle_stem}_inner_validation.csv", inner_rows)
        _write_rows(output_dir / f"{bundle_stem}_selected.csv", selected_rows)
        _write_rows(output_dir / f"{bundle_stem}_predictions.csv", prediction_rows)
        _write_rows(output_dir / f"{bundle_stem}_group_summary.csv", summary_rows)
        _write_rows(output_dir / f"{bundle_stem}_confusion.csv", bundle_confusion_rows)
        _write_rows(output_dir / f"{bundle_stem}_per_stimulus.csv", bundle_per_stimulus_rows)
        _write_rows(output_dir / f"{bundle_stem}_confusion_pairs.csv", bundle_confusion_pair_rows)

        all_outer.extend(outer_rows)
        all_inner.extend(inner_rows)
        all_selected.extend(selected_rows)
        all_predictions.extend(prediction_rows)
        all_summaries.extend(summary_rows)
        all_confusion.extend(bundle_confusion_rows)
        all_per_stimulus.extend(bundle_per_stimulus_rows)
        all_confusion_pairs.extend(bundle_confusion_pair_rows)

    _write_rows(output_dir / f"{output_stem}_outer.csv", all_outer)
    _write_rows(output_dir / f"{output_stem}_inner_validation.csv", all_inner)
    _write_rows(output_dir / f"{output_stem}_selected.csv", all_selected)
    _write_rows(output_dir / f"{output_stem}_predictions.csv", all_predictions)
    _write_rows(output_dir / f"{output_stem}_group_summary.csv", all_summaries)
    _write_rows(output_dir / f"{output_stem}_confusion.csv", all_confusion)
    _write_rows(output_dir / f"{output_stem}_per_stimulus.csv", all_per_stimulus)
    _write_rows(output_dir / f"{output_stem}_confusion_pairs.csv", all_confusion_pairs)
    return {
        "outer": all_outer,
        "inner_validation": all_inner,
        "selected": all_selected,
        "predictions": all_predictions,
        "group_summary": all_summaries,
        "confusion": all_confusion,
        "per_stimulus": all_per_stimulus,
        "confusion_pairs": all_confusion_pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing downloaded nested matrix shard artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for aggregated CSV outputs.")
    parser.add_argument("--output-stem", default="nested_matrix", help="Stem for aggregate output CSVs.")
    parser.add_argument("--signflip-permutations", type=int, default=10_000)
    parser.add_argument("--signflip-seed", type=int, default=0)
    args = parser.parse_args(argv)

    artifacts = aggregate_nested_matrix_outputs(
        args.input_dir,
        args.output_dir,
        output_stem=args.output_stem,
        signflip_permutations=args.signflip_permutations,
        signflip_seed=args.signflip_seed,
    )
    print(
        "Aggregated "
        f"{len(artifacts['outer'])} outer row(s), "
        f"{len(artifacts['selected'])} selected row(s), "
        f"and {len(artifacts['group_summary'])} bundle summary row(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
