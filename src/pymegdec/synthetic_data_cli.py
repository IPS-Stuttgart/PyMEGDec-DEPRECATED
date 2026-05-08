"""Command-line interface for generating PyMEGDec synthetic demo data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .synthetic_data import SyntheticDataConfig, write_synthetic_dataset


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in value.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one value is required.")
    return values


def _parse_float_range(value: str) -> tuple[float, float]:
    values = _parse_float_list(value)
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly two comma-separated values: start,stop.")
    return values


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """Build the synthetic-data generator parser."""

    defaults = SyntheticDataConfig()
    parser = argparse.ArgumentParser(prog=prog, description="Create private-data-free synthetic Part*Data.mat demo files.")
    parser.add_argument("--out", "--out-dir", dest="out_dir", required=True, help="Output directory for generated MAT files.")
    parser.add_argument("--participant", type=int, default=defaults.participant_id, help="Participant id used in Part* file names.")
    parser.add_argument("--classes", type=int, default=defaults.n_classes, help="Number of one-based stimulus classes.")
    parser.add_argument("--main-repeats", type=int, default=defaults.main_repeats_per_class, help="Main-experiment repeats per class.")
    parser.add_argument("--cue-repeats", type=int, default=defaults.cue_repeats_per_class, help="Cue-experiment repeats per class.")
    parser.add_argument("--channels", type=int, default=defaults.n_channels, help="Number of MEG channels.")
    parser.add_argument("--times", type=int, default=defaults.n_times, help="Number of time samples per trial.")
    parser.add_argument("--tmin", type=float, default=defaults.tmin, help="First sample time in seconds.")
    parser.add_argument("--tmax", type=float, default=defaults.tmax, help="Last sample time in seconds.")
    parser.add_argument(
        "--stimulus-window",
        type=_parse_float_range,
        default=defaults.stimulus_window,
        help="Class-informative window as start,stop in seconds.",
    )
    parser.add_argument("--signal-scale", type=float, default=defaults.signal_scale, help="Class-pattern amplitude in the stimulus window.")
    parser.add_argument("--noise-scale", type=float, default=defaults.noise_scale, help="Gaussian observation noise scale.")
    parser.add_argument("--alpha-scale", type=float, default=defaults.alpha_scale, help="Background 10 Hz carrier amplitude.")
    parser.add_argument("--cue-shift-scale", type=float, default=defaults.cue_shift_scale, help="Small cue-domain shift scale.")
    parser.add_argument("--seed", type=int, default=defaults.random_seed, help="Random seed for reproducible data.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated files.")
    parser.add_argument("--no-manifest", action="store_true", help="Do not write synthetic_data_manifest.json.")
    return parser


def make_synthetic_data(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Generate private-data-free PyMEGDec demo MAT files."""

    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    config = SyntheticDataConfig(
        participant_id=args.participant,
        n_classes=args.classes,
        main_repeats_per_class=args.main_repeats,
        cue_repeats_per_class=args.cue_repeats,
        n_channels=args.channels,
        n_times=args.times,
        tmin=args.tmin,
        tmax=args.tmax,
        stimulus_window=args.stimulus_window,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        alpha_scale=args.alpha_scale,
        cue_shift_scale=args.cue_shift_scale,
        random_seed=args.seed,
    )
    output = write_synthetic_dataset(
        args.out_dir,
        config,
        overwrite=args.overwrite,
        write_manifest=not args.no_manifest,
    )
    print(f"Wrote main data: {output.main_path}")
    print(f"Wrote cue data: {output.cue_path}")
    if output.manifest_path is not None:
        print(f"Wrote manifest: {output.manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``pymegdec-make-synthetic-data``."""

    return make_synthetic_data(argv)


if __name__ == "__main__":
    raise SystemExit(main())
