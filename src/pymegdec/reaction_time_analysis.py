"""Reaction-time association analysis for exploratory alpha metrics."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymegdec.alpha_metrics import (
    AlphaMetricConfig,
    compute_alpha_metrics,
    count_trials,
    load_participant_data,
    write_alpha_metrics_csv,
)
from pymegdec.alpha_signal import get_data_field
from pymegdec.data_config import resolve_data_folder
from scipy import stats

DEFAULT_ALPHA_RT_METRICS = (
    "log_alpha_power",
    "phase_concentration",
    "phase_plane_fit",
    "spatial_freq_rad_per_mm",
    "speed_m_per_s",
    "gradient_x",
    "gradient_y",
    "direction_sin",
    "direction_cos",
)

REACTION_TIME_FIELD_CANDIDATES = (
    "reaction_time",
    "reaction_time_s",
    "response_time",
    "response_time_s",
    "rt",
)

TRIAL_INDEX_BASE_CHOICES = (0, 1)


class ReactionTimeUnavailableError(ValueError):
    """Raised when reaction times are not present in the available metadata."""


@dataclass(frozen=True)
class ReactionTimeCsvConfig:
    """Column mapping for an external reaction-time CSV.

    ``trial_index_base`` describes the CSV's trial numbering. Alpha rows use
    zero-based trial indices; pass ``trial_index_base=1`` for behavioral CSVs
    numbered 1..N.
    """

    participant_column: str | None = None
    trial_column: str | None = None
    reaction_time_column: str | None = None
    dataset_column: str | None = None
    default_participant: int | str | None = None
    default_dataset: str = "main"
    reaction_time_scale: float = 1.0
    trial_index_base: int = 0


@dataclass(frozen=True)
class AlphaReactionTimeExportConfig:  # pylint: disable=too-many-instance-attributes
    """Inputs and outputs for an alpha/RT export run."""

    reaction_times_path: str | Path | None = None
    alpha_metrics_path: str | Path | None = None
    joined_output_path: str | Path | None = None
    summary_output_path: str | Path | None = None
    cue: bool = False
    alpha_config: AlphaMetricConfig | None = None
    csv_config: ReactionTimeCsvConfig | None = None
    trialinfo_rt_column: int | None = None
    metrics: tuple[str, ...] = DEFAULT_ALPHA_RT_METRICS


def parse_participant_spec(spec):
    """Parse participant specs like ``1-4,6,8``."""

    participants: list[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = token.split("-", maxsplit=1)
            participants.extend(range(int(start), int(stop) + 1))
        else:
            participants.append(int(token))
    return sorted(set(participants))


def load_csv_rows(path):
    """Load a CSV file as dictionaries."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv_rows(rows, output_path):
    """Write dictionary rows to ``output_path``."""

    write_alpha_metrics_csv(rows, output_path)


def _clean_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _to_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _to_int(value):
    return int(float(str(value).strip()))


def _validate_trial_index_base(trial_index_base):
    if trial_index_base not in TRIAL_INDEX_BASE_CHOICES:
        raise ValueError(f"trial_index_base must be one of {TRIAL_INDEX_BASE_CHOICES}, got {trial_index_base!r}.")
    return trial_index_base


def _normalize_csv_trial(value, trial_index_base):
    raw_trial = _to_int(value)
    zero_based_trial = raw_trial - _validate_trial_index_base(trial_index_base)
    if zero_based_trial < 0:
        raise ValueError(
            f"CSV trial value {raw_trial!r} with trial_index_base={trial_index_base} "
            "maps to a negative zero-based trial index."
        )
    return zero_based_trial


def _column(fieldnames, explicit, candidates, *, required=True):
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"CSV column {explicit!r} was not found.")
        return explicit

    lookup = {field_name.lower(): field_name for field_name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(f"CSV must contain one of these columns: {', '.join(candidates)}.")
    return None


def load_reaction_time_csv(path, config=None):
    """Load external reaction times and normalize key columns."""

    config = config or ReactionTimeCsvConfig()
    trial_index_base = _validate_trial_index_base(config.trial_index_base)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        participant_column = _column(
            fieldnames,
            config.participant_column,
            ("participant", "participant_id", "part"),
            required=config.default_participant is None,
        )
        trial_column = _column(fieldnames, config.trial_column, ("trial", "trial_idx", "trial_index"))
        rt_column = _column(fieldnames, config.reaction_time_column, REACTION_TIME_FIELD_CANDIDATES)
        dataset_column = _column(
            fieldnames,
            config.dataset_column,
            ("dataset", "condition", "source"),
            required=False,
        )

        rows = []
        for row in reader:
            rows.append(
                {
                    "participant": _clean_id(row[participant_column] if participant_column else config.default_participant),
                    "dataset": (row[dataset_column] if dataset_column else config.default_dataset),
                    "trial": _normalize_csv_trial(row[trial_column], trial_index_base),
                    "reaction_time": _to_float(row[rt_column]) * config.reaction_time_scale,
                }
            )
    return rows


def _data_field_names(data):
    if isinstance(data, dict):
        return tuple(data.keys())
    return tuple(data.dtype.names or ())


def _has_field(data, field_name):
    return field_name in _data_field_names(data)


def _trialinfo_matrix(data):
    trialinfo = np.asarray(get_data_field(data, "trialinfo"))
    n_trials = count_trials(data)
    if trialinfo.ndim == 1:
        return trialinfo.reshape(-1, 1)
    if trialinfo.shape[0] == n_trials:
        return trialinfo
    if trialinfo.shape[1] == n_trials:
        return trialinfo.T
    raise ValueError(f"Cannot align trialinfo shape {trialinfo.shape} to {n_trials} trials.")


def extract_reaction_times_from_data(
    data,
    *,
    participant_id=None,
    dataset="main",
    trialinfo_rt_column=None,
    reaction_time_scale=1.0,
):
    """Extract reaction times from MAT metadata when such a field is present."""

    n_trials = count_trials(data)
    for field_name in REACTION_TIME_FIELD_CANDIDATES:
        if _has_field(data, field_name):
            values = np.asarray(get_data_field(data, field_name), dtype=float).ravel()
            return _reaction_time_rows(values, n_trials, participant_id, dataset, reaction_time_scale)

    if trialinfo_rt_column is not None:
        trialinfo = _trialinfo_matrix(data)
        if trialinfo_rt_column >= trialinfo.shape[1]:
            raise ValueError(f"trialinfo column {trialinfo_rt_column} does not exist.")
        return _reaction_time_rows(
            trialinfo[:, trialinfo_rt_column],
            n_trials,
            participant_id,
            dataset,
            reaction_time_scale,
        )

    raise ReactionTimeUnavailableError("Reaction times were not found in the MAT data. Provide an external " "reaction-time CSV or pass a trialinfo RT column if one is present.")


def _reaction_time_rows(values, n_trials, participant_id, dataset, reaction_time_scale):
    if values.size != n_trials:
        raise ValueError(f"Expected {n_trials} reaction times, got {values.size}.")
    return [
        {
            "participant": _clean_id(participant_id),
            "dataset": dataset,
            "trial": trial_idx,
            "reaction_time": float(values[trial_idx]) * reaction_time_scale,
        }
        for trial_idx in range(n_trials)
    ]


def extract_reaction_times_for_participants(
    data_folder,
    participants,
    *,
    cue=False,
    trialinfo_rt_column=None,
    reaction_time_scale=1.0,
):
    """Extract reaction times for participants from MAT data metadata."""

    rows = []
    dataset = "cue" if cue else "main"
    for participant_id in participants:
        data = load_participant_data(data_folder, participant_id, cue=cue)
        rows.extend(
            extract_reaction_times_from_data(
                data,
                participant_id=participant_id,
                dataset=dataset,
                trialinfo_rt_column=trialinfo_rt_column,
                reaction_time_scale=reaction_time_scale,
            )
        )
    return rows


def compute_alpha_rows_for_participants(data_folder, participants, *, cue=False, config=None):
    """Compute alpha metrics for participants without writing intermediate files."""

    rows = []
    config = config or AlphaMetricConfig()
    dataset = "cue" if cue else "main"
    for participant_id in participants:
        data = load_participant_data(data_folder, participant_id, cue=cue)
        rows.extend(compute_alpha_metrics(data, participant_id=participant_id, dataset=dataset, config=config))
    return rows


def load_participant_alpha_rows(data_folder, participants, *, cue=False, alpha_metrics_path=None, config=None):
    """Load precomputed alpha rows or compute them from participant MAT files."""

    if alpha_metrics_path:
        return load_csv_rows(alpha_metrics_path)
    data_folder = resolve_data_folder(data_folder)
    return compute_alpha_rows_for_participants(data_folder, participants, cue=cue, config=config)


def load_participant_reaction_time_rows(
    data_folder,
    participants,
    *,
    cue=False,
    reaction_times_path=None,
    csv_config=None,
    trialinfo_rt_column=None,
):
    """Load external reaction times or extract them from participant MAT files."""

    if reaction_times_path:
        return load_reaction_time_csv(reaction_times_path, csv_config)
    data_folder = resolve_data_folder(data_folder)
    scale = 1.0 if csv_config is None else csv_config.reaction_time_scale
    return extract_reaction_times_for_participants(
        data_folder,
        participants,
        cue=cue,
        trialinfo_rt_column=trialinfo_rt_column,
        reaction_time_scale=scale,
    )


def _join_key(row):
    return (
        _clean_id(row.get("participant")),
        str(row.get("dataset", "main")),
        _to_int(row.get("trial")),
    )


def _trial_group_key(row):
    return (
        _clean_id(row.get("participant")),
        str(row.get("dataset", "main")),
    )


def _group_trials_by_participant_dataset(rows):
    grouped: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        grouped.setdefault(_trial_group_key(row), set()).add(_to_int(row.get("trial")))
    return grouped


def _raise_if_likely_one_based_reaction_trials(alpha_rows, reaction_time_rows):
    """Raise when RT rows look like unconverted one-based trial numbers."""

    alpha_trials_by_group = _group_trials_by_participant_dataset(alpha_rows)
    reaction_trials_by_group = _group_trials_by_participant_dataset(reaction_time_rows)
    for participant, dataset in sorted(alpha_trials_by_group.keys() & reaction_trials_by_group.keys()):
        alpha_trials = alpha_trials_by_group[(participant, dataset)]
        reaction_trials = reaction_trials_by_group[(participant, dataset)]
        if not alpha_trials or not reaction_trials:
            continue

        current_matches = len(alpha_trials & reaction_trials)
        one_based_matches = len(alpha_trials & {trial - 1 for trial in reaction_trials})
        max_possible_matches = min(len(alpha_trials), len(reaction_trials))
        if one_based_matches > current_matches and one_based_matches >= max_possible_matches:
            raise ValueError(
                "Reaction-time trial numbers for "
                f"participant {participant!r}, dataset {dataset!r} look one-based. "
                "PyMEGDec joins against zero-based alpha trial indices. Pass "
                "--reaction-time-trial-base 1, or set "
                "ReactionTimeCsvConfig(trial_index_base=1), so the CSV trial "
                "column is converted before joining."
            )


def join_alpha_reaction_times(alpha_rows, reaction_time_rows):
    """Join alpha metric rows with reaction times by participant, dataset, and trial."""

    alpha_rows = list(alpha_rows)
    reaction_time_rows = list(reaction_time_rows)
    _raise_if_likely_one_based_reaction_trials(alpha_rows, reaction_time_rows)

    reaction_by_key = {}
    for row in reaction_time_rows:
        key = _join_key(row)
        if key in reaction_by_key:
            raise ValueError(f"Duplicate reaction-time row for key {key}.")
        reaction_by_key[key] = row

    joined_rows = []
    for alpha_row in alpha_rows:
        reaction_row = reaction_by_key.get(_join_key(alpha_row))
        if reaction_row is None:
            continue
        joined_row = dict(alpha_row)
        joined_row["reaction_time"] = reaction_row["reaction_time"]
        direction = _to_float(joined_row.get("direction_rad"))
        joined_row["direction_sin"] = math.sin(direction) if np.isfinite(direction) else np.nan
        joined_row["direction_cos"] = math.cos(direction) if np.isfinite(direction) else np.nan
        joined_rows.append(joined_row)

    if not joined_rows:
        raise ValueError("No alpha rows matched the reaction-time rows.")
    return joined_rows


def _finite_metric_arrays(rows, metric):
    x_values = np.array([_to_float(row.get(metric)) for row in rows], dtype=float)
    y_values = np.array([_to_float(row.get("reaction_time")) for row in rows], dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]


def _association_row(scope, participant, metric, rows, min_trials):
    x_values, y_values = _finite_metric_arrays(rows, metric)
    result = _empty_association(scope, participant, metric, x_values, y_values)
    if x_values.size < min_trials or np.ptp(x_values) == 0 or np.ptp(y_values) == 0:
        return result

    pearson = stats.pearsonr(x_values, y_values)
    regression = stats.linregress(x_values, y_values)
    result.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "slope_reaction_time_per_unit": float(regression.slope),
            "intercept_reaction_time": float(regression.intercept),
        }
    )
    return result


def _empty_association(scope, participant, metric, x_values, y_values):
    return {
        "scope": scope,
        "participant": participant,
        "metric": metric,
        "n_trials": int(x_values.size),
        "metric_mean": float(np.mean(x_values)) if x_values.size else np.nan,
        "reaction_time_mean": float(np.mean(y_values)) if y_values.size else np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "slope_reaction_time_per_unit": np.nan,
        "intercept_reaction_time": np.nan,
    }


def _group_by_participant(rows):
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_clean_id(row.get("participant")), []).append(row)
    return grouped


def _within_participant_centered_rows(grouped_rows, metric):
    centered_rows = []
    for participant_rows in grouped_rows.values():
        x_values, y_values = _finite_metric_arrays(participant_rows, metric)
        for x_value, y_value in zip(x_values, y_values):
            centered_rows.append(
                {
                    "participant": "",
                    metric: x_value - np.mean(x_values),
                    "reaction_time": y_value - np.mean(y_values),
                }
            )
    return centered_rows


def analyze_alpha_reaction_times(rows, metrics=DEFAULT_ALPHA_RT_METRICS, min_trials=3):
    """Compute per-participant and within-participant pooled alpha/RT associations."""

    summary_rows = []
    grouped_rows = _group_by_participant(rows)
    for metric in metrics:
        for participant, participant_rows in grouped_rows.items():
            summary_rows.append(_association_row("participant", participant, metric, participant_rows, min_trials))
        centered_rows = _within_participant_centered_rows(grouped_rows, metric)
        summary_rows.append(_association_row("pooled_within_participant", "", metric, centered_rows, min_trials))
    return summary_rows


def write_alpha_reaction_time_plots(rows, output_dir, metrics=DEFAULT_ALPHA_RT_METRICS, min_trials=3):
    """Write simple scatter plots for joined alpha/RT rows."""

    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for metric in metrics:
        x_values, y_values = _finite_metric_arrays(rows, metric)
        if x_values.size < min_trials:
            continue
        figure, axes = plt.subplots(figsize=(5, 4))
        axes.scatter(x_values, y_values, alpha=0.6, s=18)
        axes.set_xlabel(metric)
        axes.set_ylabel("reaction_time")
        figure.tight_layout()
        output_path = output_dir / f"alpha_rt_{metric}.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def export_alpha_reaction_time_analysis(
    data_folder,
    participants,
    config=None,
):
    """Load alpha and RT data, write joined trial rows and summary associations."""

    config = config or AlphaReactionTimeExportConfig()
    alpha_rows = load_participant_alpha_rows(
        data_folder,
        participants,
        cue=config.cue,
        alpha_metrics_path=config.alpha_metrics_path,
        config=config.alpha_config,
    )
    reaction_time_rows = load_participant_reaction_time_rows(
        data_folder,
        participants,
        cue=config.cue,
        reaction_times_path=config.reaction_times_path,
        csv_config=config.csv_config,
        trialinfo_rt_column=config.trialinfo_rt_column,
    )
    joined_rows = join_alpha_reaction_times(alpha_rows, reaction_time_rows)
    summary_rows = analyze_alpha_reaction_times(joined_rows, metrics=config.metrics)

    if config.joined_output_path:
        write_csv_rows(joined_rows, config.joined_output_path)
    if config.summary_output_path:
        write_csv_rows(summary_rows, config.summary_output_path)
    return joined_rows, summary_rows


def available_participants(data_folder, *, cue=False):
    """Return participant ids with matching MAT files in ``data_folder``."""

    data_folder = resolve_data_folder(data_folder)
    suffix = "CueData" if cue else "Data"
    participants = []
    for path in Path(data_folder).glob(f"Part*{suffix}.mat"):
        participant = path.name.removeprefix("Part").removesuffix(f"{suffix}.mat")
        if participant.isdigit():
            participants.append(int(participant))
    return sorted(participants)


# Route reusable reaction-time operations through NeuRepTrace when the upstream
# module is available.  PyMEGDec keeps only the alpha-specific orchestration,
# plotting, and MAT-file glue above; the generic CSV parsing, trial-index
# normalization, row joins, and metric/RT association summaries belong in
# NeuRepTrace.
try:  # pragma: no cover - exercised once NeuRepTrace carries the upstream helper
    from neureptrace.behavior import reaction_time as _reptrace_rt
except ImportError:  # pragma: no cover - fallback keeps historical checkouts usable
    pass
else:
    ReactionTimeUnavailableError = _reptrace_rt.ReactionTimeUnavailableError  # type: ignore[assignment, no-redef]
    ReactionTimeCsvConfig = _reptrace_rt.ReactionTimeCsvConfig  # type: ignore[assignment, no-redef]
    REACTION_TIME_FIELD_CANDIDATES = _reptrace_rt.REACTION_TIME_FIELD_CANDIDATES
    TRIAL_INDEX_BASE_CHOICES = _reptrace_rt.TRIAL_INDEX_BASE_CHOICES

    _clean_id = _reptrace_rt._clean_id
    _to_float = _reptrace_rt._to_float
    _to_int = _reptrace_rt._to_int
    _validate_trial_index_base = _reptrace_rt._validate_trial_index_base
    _normalize_csv_trial = _reptrace_rt._normalize_csv_trial
    _raise_if_likely_one_based_reaction_trials = _reptrace_rt._raise_if_likely_one_based_reaction_trials

    load_reaction_time_csv = _reptrace_rt.load_reaction_time_csv  # type: ignore[assignment]

    def _reaction_time_rows(values, n_trials, participant_id, dataset, reaction_time_scale):  # type: ignore[no-redef]
        values = np.asarray(values, dtype=float).ravel()
        if values.size != n_trials:
            raise ValueError(f"Expected {n_trials} reaction times, got {values.size}.")
        return _reptrace_rt.reaction_time_rows_from_values(
            values,
            participant=participant_id,
            dataset=dataset,
            reaction_time_scale=reaction_time_scale,
        )

    def join_alpha_reaction_times(alpha_rows, reaction_time_rows):  # type: ignore[no-redef]
        """Join alpha rows with RTs via NeuRepTrace and add alpha direction columns."""

        joined_rows = _reptrace_rt.join_reaction_times(alpha_rows, reaction_time_rows)
        for joined_row in joined_rows:
            direction = _to_float(joined_row.get("direction_rad"))
            joined_row["direction_sin"] = math.sin(direction) if np.isfinite(direction) else np.nan
            joined_row["direction_cos"] = math.cos(direction) if np.isfinite(direction) else np.nan
        return joined_rows

    def analyze_alpha_reaction_times(rows, metrics=DEFAULT_ALPHA_RT_METRICS, min_trials=3):  # type: ignore[no-redef]
        """Compute alpha/RT associations through NeuRepTrace's generic summarizer."""

        return _reptrace_rt.analyze_metric_reaction_times(rows, metrics, min_trials=min_trials)


    __all__ = [
        "DEFAULT_ALPHA_RT_METRICS",
        "AlphaReactionTimeExportConfig",
        "ReactionTimeCsvConfig",
        "ReactionTimeUnavailableError",
        "analyze_alpha_reaction_times",
        "available_participants",
        "export_alpha_reaction_time_analysis",
        "extract_reaction_times_from_data",
        "join_alpha_reaction_times",
        "load_reaction_time_csv",
        "parse_participant_spec",
        "write_alpha_reaction_time_plots",
        "write_csv_rows",
    ]
