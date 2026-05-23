"""Compatibility wrapper for NeuRepTrace-owned PyMEGDec dataset specs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_ENV_VAR = "PYMEGDEC_DATA_DIR"
DEFAULT_DATASET_ID = "bushmeg"

# Keep PyMEGDec usable with older NeuRepTrace checkouts, but prefer the
# maintained helper as soon as it is available.  The duplicated fallback can be
# removed once PyMEGDec's NeuRepTrace dependency is pinned beyond the migration
# commit that introduced neureptrace.compat.pymegdec_dataset_spec.
try:  # pragma: no cover - exercised when the matching NeuRepTrace commit is installed
    from neureptrace.compat.pymegdec_dataset_spec import (
        build_pymegdec_bushmeg_dataset_spec_text as _build_pymegdec_bushmeg_dataset_spec_text,
        write_pymegdec_bushmeg_dataset_spec as _write_pymegdec_bushmeg_dataset_spec,
    )
except ImportError:  # pragma: no cover - fallback keeps historical environments usable
    _build_pymegdec_bushmeg_dataset_spec_text = None
    _write_pymegdec_bushmeg_dataset_spec = None

_FALLBACK_TEMPLATE = """schema_version: neureptrace.dataset.v1
dataset_id: {dataset_id}
description: PyMEGDec-style MEG participant files described declaratively.

root:
{root_path_block}  env: {env_var}
  fallback_file: .pymegdec-data-dir

subjects:
  include: "{participants}"

splits:
  main:
    loader: matlab_fieldtrip
    path_template: "Part{{subject}}Data.mat"
    mat_key: data
    trial_key: trial
    time_key: time
    channel_key: label
    label_key: trialinfo
    label_index_base: 1
    trial_layout: channels_by_time

  cue:
    loader: matlab_fieldtrip
    path_template: "Part{{subject}}CueData.mat"
    mat_key: data
    trial_key: trial
    time_key: time
    channel_key: label
    label_key: trialinfo
    label_index_base: 1
    trial_layout: channels_by_time

labels:
  chance_classes: 16
  index_base: 1
  subtract_one_when_no_null_class: true

preprocessing_defaults:
  frequency_range_hz: [0.0, .inf]
  window_size_s: 0.1
  train_window_center_s: 0.2
  null_window_center_s: null
  resample_hz: null
  pca_components: 100

workflows:
  stimulus_transfer:
    split: main
    manifest:
      paired_split: cue
      transfer_direction: main-to-cue
      classifier: multiclass-svm
      chance: 0.0625
      window_start_s: -0.2
      window_stop_s: 0.6
      window_step_s: 0.05

  stimulus_transfer_reverse:
    split: cue
    manifest:
      paired_split: main
      transfer_direction: cue-to-main
      classifier: multiclass-svm
      chance: 0.0625
      window_start_s: -0.2
      window_stop_s: 0.6
      window_step_s: 0.05

outputs:
  default_dir: outputs
"""


def _fallback_build_spec_text(
    *,
    participants: str = DEFAULT_PARTICIPANTS,
    env_var: str = DEFAULT_ENV_VAR,
    data_dir: str | Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> str:
    root_path_block = ""
    if data_dir is not None:
        root_path_block = f"  path: {json.dumps(str(data_dir))}\n"
    return _FALLBACK_TEMPLATE.format(
        dataset_id=dataset_id,
        participants=participants,
        env_var=env_var,
        root_path_block=root_path_block,
    )


def build_neureptrace_dataset_spec_text(
    *,
    participants: str = DEFAULT_PARTICIPANTS,
    env_var: str = DEFAULT_ENV_VAR,
    data_dir: str | Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> str:
    """Return a YAML NeuRepTrace dataset spec for PyMEGDec-style files."""

    if _build_pymegdec_bushmeg_dataset_spec_text is not None:
        return _build_pymegdec_bushmeg_dataset_spec_text(
            participants=participants,
            env_var=env_var,
            data_dir=data_dir,
            dataset_id=dataset_id,
        )
    return _fallback_build_spec_text(
        participants=participants,
        env_var=env_var,
        data_dir=data_dir,
        dataset_id=dataset_id,
    )


build_pymegdec_bushmeg_dataset_spec_text = build_neureptrace_dataset_spec_text


def write_neureptrace_dataset_spec(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Write a NeuRepTrace YAML dataset spec for the historical PyMEGDec file convention."""

    if _write_pymegdec_bushmeg_dataset_spec is not None:
        return _write_pymegdec_bushmeg_dataset_spec(argv, prog=prog)

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Write a NeuRepTrace YAML dataset spec for the historical PyMEGDec Part*Data.mat convention.",
    )
    parser.add_argument("--out", type=Path, default=Path("configs/bushmeg.yml"), help="Output YAML path.")
    parser.add_argument("--participants", default=DEFAULT_PARTICIPANTS, help="Participant ids, for example 1-4,6,8.")
    parser.add_argument("--env-var", default=DEFAULT_ENV_VAR, help="Environment variable used by root.env.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Dataset id written into the spec.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional explicit data root to write into root.path. If omitted, the spec uses env/fallback root resolution.",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_neureptrace_dataset_spec_text(
            participants=args.participants,
            env_var=args.env_var,
            data_dir=args.data_dir,
            dataset_id=args.dataset_id,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print("Validate with: neureptrace dataset validate", args.out)
    return 0


write_pymegdec_bushmeg_dataset_spec = write_neureptrace_dataset_spec

__all__ = [
    "DEFAULT_DATASET_ID",
    "DEFAULT_ENV_VAR",
    "DEFAULT_PARTICIPANTS",
    "build_neureptrace_dataset_spec_text",
    "build_pymegdec_bushmeg_dataset_spec_text",
    "write_neureptrace_dataset_spec",
    "write_pymegdec_bushmeg_dataset_spec",
]


if __name__ == "__main__":
    raise SystemExit(write_neureptrace_dataset_spec())
