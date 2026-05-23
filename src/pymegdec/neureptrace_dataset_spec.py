"""Compatibility wrapper for NeuRepTrace-owned PyMEGDec/BUSH-MEG specs."""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_PARTICIPANTS = "1-4,6,8,9,10,13-27"
DEFAULT_ENV_VAR = "PYMEGDEC_DATA_DIR"
DEFAULT_DATASET_ID = "bushmeg"

try:
    from neureptrace.datasets import pymegdec as _neureptrace_pymegdec
except ModuleNotFoundError as exc:
    if exc.name not in {
        "neureptrace",
        "neureptrace.datasets",
        "neureptrace.datasets.pymegdec",
    }:
        raise
    _neureptrace_pymegdec = None
else:
    DEFAULT_PARTICIPANTS = getattr(_neureptrace_pymegdec, "DEFAULT_PARTICIPANTS", DEFAULT_PARTICIPANTS)
    DEFAULT_ENV_VAR = getattr(_neureptrace_pymegdec, "DEFAULT_ENV_VAR", DEFAULT_ENV_VAR)

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


def _has_parameter(function: Callable[..., Any], parameter: str) -> bool:
    try:
        return parameter in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _call_with_supported_keywords(function: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return function(**supported)


def _neureptrace_spec_builder() -> Callable[..., str] | None:
    if _neureptrace_pymegdec is None:
        return None
    return getattr(
        _neureptrace_pymegdec,
        "build_neureptrace_dataset_spec_text",
        getattr(_neureptrace_pymegdec, "build_pymegdec_bushmeg_dataset_spec_text", None),
    )


def build_neureptrace_dataset_spec_text(
    *,
    participants: str = DEFAULT_PARTICIPANTS,
    env_var: str = DEFAULT_ENV_VAR,
    data_dir: str | Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> str:
    """Return the NeuRepTrace-owned YAML spec for PyMEGDec-style files."""

    builder = _neureptrace_spec_builder()
    if builder is None or (dataset_id != DEFAULT_DATASET_ID and not _has_parameter(builder, "dataset_id")):
        return _fallback_build_spec_text(
            participants=participants,
            env_var=env_var,
            data_dir=data_dir,
            dataset_id=dataset_id,
        )
    return _call_with_supported_keywords(
        builder,
        participants=participants,
        env_var=env_var,
        data_dir=data_dir,
        dataset_id=dataset_id,
    )


build_pymegdec_bushmeg_spec_text = build_neureptrace_dataset_spec_text


def write_neureptrace_dataset_spec(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Write the NeuRepTrace-owned PyMEGDec/BUSH-MEG YAML dataset spec."""

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


if __name__ == "__main__":
    raise SystemExit(write_neureptrace_dataset_spec())


__all__ = [
    "DEFAULT_DATASET_ID",
    "DEFAULT_ENV_VAR",
    "DEFAULT_PARTICIPANTS",
    "build_neureptrace_dataset_spec_text",
    "build_pymegdec_bushmeg_spec_text",
    "write_neureptrace_dataset_spec",
]
