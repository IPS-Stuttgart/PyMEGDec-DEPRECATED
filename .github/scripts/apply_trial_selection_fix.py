from __future__ import annotations

from pathlib import Path
import subprocess


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def insert_after(text: str, anchor: str, insert: str) -> str:
    if insert.strip() in text:
        return text
    return text.replace(anchor, anchor + insert, 1)


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Missing replacement anchor:\n{old}")
    return text.replace(old, new, 1)


def replace_after(text: str, marker: str, old: str, new: str) -> str:
    start = text.index(marker)
    pos = text.index(old, start)
    if text[pos : pos + len(new)] == new:
        return text
    return text[:pos] + new + text[pos + len(old) :]


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    if replacement.strip() in text:
        return text
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement + text[end:]


core_path = "src/pymegdec/_stimulus_cross_subject_core.py"
core = read(core_path)
core = insert_after(
    core,
    'DEFAULT_CROSS_SUBJECT_SELECTION_METRIC = "balanced_accuracy"\n',
    'DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"\n'
    'DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0\n'
    'TRIAL_SELECTION_MODES = ("random", "first")\n',
)
core = replace_once(
    core,
    'FEATURE_MODES = ("sensor_mean", "sensor_flat")\nDEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"\n',
    'DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"\n',
) if 'FEATURE_MODES = ("sensor_mean", "sensor_flat")\nDEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"\n' in core else core
core = replace_once(
    core,
    'DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0\nTRIAL_SELECTION_MODES = ("random", "first")\nFEATURE_MODES = ("sensor_mean", "sensor_flat")\n',
    'FEATURE_MODES = ("sensor_mean", "sensor_flat")\nDEFAULT_CROSS_SUBJECT_TRIAL_SELECTION = "random"\nDEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0\nTRIAL_SELECTION_MODES = ("random", "first")\n',
) if 'DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED = 0\nTRIAL_SELECTION_MODES = ("random", "first")\nFEATURE_MODES = ("sensor_mean", "sensor_flat")\n' in core else core
core = replace_once(
    core,
    '    "max_trials_per_class_per_participant",\n    "label_shuffle_control",\n',
    '    "max_trials_per_class_per_participant",\n    "trial_selection",\n    "trial_selection_seed",\n    "label_shuffle_control",\n',
)
core = replace_once(
    core,
    '    max_trials_per_class_per_participant: int | None = None\n    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES\n',
    '    max_trials_per_class_per_participant: int | None = None\n    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION\n    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED\n    chance_classes: int = DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES\n',
)
core = replace_once(
    core,
    '    n_baseline_samples: int\n    max_trials_per_class_per_participant: int | None\n',
    '    n_baseline_samples: int\n    max_trials_per_class_per_participant: int | None\n    trial_indices: np.ndarray | None = None\n    trial_selection: str = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION\n    trial_selection_seed: int | None = DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED\n',
)
core = replace_once(
    core,
    '    max_trials_per_class_per_participant=None,\n    chance_classes=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,\n',
    '    max_trials_per_class_per_participant=None,\n    trial_selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,\n    trial_selection_seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,\n    chance_classes=DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,\n',
)
core = replace_once(
    core,
    '            max_trials_per_class_per_participant=max_trials_per_class_per_participant,\n            chance_classes=chance_classes,\n',
    '            max_trials_per_class_per_participant=max_trials_per_class_per_participant,\n            trial_selection=trial_selection,\n            trial_selection_seed=trial_selection_seed,\n            chance_classes=chance_classes,\n',
)
core = replace_once(
    core,
    '    trial_indices = _selected_trial_indices(all_labels, config.max_trials_per_class_per_participant)\n',
    '    trial_indices = _selected_trial_indices(\n        all_labels,\n        config.max_trials_per_class_per_participant,\n        selection=config.trial_selection,\n        seed=config.trial_selection_seed,\n        participant=participant,\n    )\n',
)
core = replace_after(
    core,
    'return ParticipantFeatureSet(',
    '        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,\n',
    '        max_trials_per_class_per_participant=config.max_trials_per_class_per_participant,\n        trial_indices=np.asarray(trial_indices, dtype=int),\n        trial_selection=config.trial_selection,\n        trial_selection_seed=config.trial_selection_seed,\n',
)
core = replace_after(
    core,
    'def summarize_cross_subject_stimulus_smoke',
    '            "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,\n',
    '            "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,\n            "trial_selection": config.trial_selection,\n            "trial_selection_seed": _seed_field(config.trial_selection_seed),\n',
)
core = insert_after(
    core,
    '    trial_cap_counts = Counter(str(row["max_trials_per_class_per_participant"]) for row in outer_rows)\n',
    '    trial_selection_counts = _row_value_counts(outer_rows, "selected_trial_selection", fallback_key="trial_selection")\n'
    '    trial_selection_seed = _single_row_value(outer_rows, "trial_selection_seed", default="")\n',
)
core = replace_after(
    core,
    'def summarize_nested_cross_subject_stimulus',
    '            "max_trials_per_class_per_participant_counts": _format_counter(trial_cap_counts),\n',
    '            "max_trials_per_class_per_participant_counts": _format_counter(trial_cap_counts),\n            "trial_selection_counts": _format_counter(trial_selection_counts),\n            "trial_selection_seed": trial_selection_seed,\n',
)
core = replace_after(
    core,
    'def _feature_cache_key',
    '        config.max_trials_per_class_per_participant,\n',
    '        config.max_trials_per_class_per_participant,\n        str(config.trial_selection),\n        _seed_field(config.trial_selection_seed),\n',
)
core = replace_after(
    core,
    'def _select_nested_candidate',
    '                "selected_max_trials_per_class_per_participant": example["max_trials_per_class_per_participant"],\n',
    '                "selected_max_trials_per_class_per_participant": example["max_trials_per_class_per_participant"],\n                "selected_trial_selection": example["trial_selection"],\n                "selected_trial_selection_seed": example.get("trial_selection_seed", ""),\n',
)
core = replace_after(
    core,
    'def _score_outer_fold_model',
    '        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,\n',
    '        "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,\n        "trial_selection": config.trial_selection,\n        "trial_selection_seed": _seed_field(config.trial_selection_seed),\n',
)
core = replace_once(
    core,
    'def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):\n    train_window = _centered_window(config.window_center, config.window_size)\n    rows = []\n    for trial_idx, (true_label, predicted_label, true_label_rank) in enumerate(zip(test_labels, predictions, true_label_ranks)):\n',
    'def _prediction_rows(test_set, test_labels, predictions, true_label_ranks, *, config, actual_components_pca):\n    train_window = _centered_window(config.window_center, config.window_size)\n    trial_indices = _feature_set_trial_indices(test_set)\n    rows = []\n    for trial_idx, true_label, predicted_label, true_label_rank in zip(trial_indices, test_labels, predictions, true_label_ranks):\n',
)
core = replace_after(
    core,
    'def _prediction_rows',
    '                "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,\n',
    '                "max_trials_per_class_per_participant": config.max_trials_per_class_per_participant,\n                "trial_selection": config.trial_selection,\n                "trial_selection_seed": _seed_field(config.trial_selection_seed),\n',
)
core = replace_between(
    core,
    'def _selected_trial_indices',
    'def _iter_trial_indices',
    '''def _selected_trial_indices(
    labels,
    max_trials_per_class,
    *,
    selection=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
    seed=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
    participant=None,
):
    labels = np.asarray(labels).ravel()
    if max_trials_per_class is None:
        return np.arange(labels.shape[0], dtype=int)
    max_trials_per_class = int(max_trials_per_class)
    if max_trials_per_class <= 0:
        raise ValueError("max_trials_per_class_per_participant must be positive.")
    selection = _normalize_trial_selection(selection)

    if selection == "first":
        selected = []
        counts: Counter[int] = Counter()
        for index, label in enumerate(labels):
            if counts[int(label)] < max_trials_per_class:
                selected.append(index)
                counts[int(label)] += 1
        return np.asarray(selected, dtype=int)

    if selection == "random":
        rng = _trial_selection_rng(seed, participant)
        selected = []
        for label in np.unique(labels):
            class_indices = np.flatnonzero(labels == label)
            if class_indices.size > max_trials_per_class:
                class_indices = rng.choice(class_indices, size=max_trials_per_class, replace=False)
            selected.extend(int(index) for index in class_indices)
        return np.asarray(sorted(selected), dtype=int)

    raise ValueError(f"Unsupported trial selection policy: {selection}")


def _trial_selection_rng(seed, participant):
    if seed is None:
        return np.random.default_rng()
    seed_values = [int(seed)]
    if participant is not None:
        seed_values.append(int(participant))
    return np.random.default_rng(np.random.SeedSequence(seed_values))


def _feature_set_trial_indices(feature_set):
    trial_indices = getattr(feature_set, "trial_indices", None)
    if trial_indices is None:
        return np.arange(np.asarray(feature_set.labels).shape[0], dtype=int)
    return np.asarray(trial_indices, dtype=int).ravel()


''',
)
core = insert_after(
    core,
    'def _one_sided_signflip_p_value(differences, *, n_permutations, seed):\n    differences = np.asarray(differences, dtype=float)\n    differences = differences[np.isfinite(differences)]\n    if differences.size == 0:\n        return np.nan\n    observed = float(np.mean(differences))\n    if observed <= 0:\n        return 1.0\n    if differences.size <= 16:\n        exact_signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * differences.size)).T.reshape(-1, differences.size)\n        null_means = exact_signs @ differences / differences.size\n        return float(np.mean(null_means >= observed))\n    rng = np.random.default_rng(seed)\n    random_signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_permutations), differences.size))\n    null_means = random_signs @ differences / differences.size\n    return float((np.sum(null_means >= observed) + 1) / (int(n_permutations) + 1))\n\n\n',
    'def _seed_field(seed):\n    return "" if seed is None else int(seed)\n\n\n',
)
core = replace_after(
    core,
    'def _normalized_config',
    '        max_trials_per_class_per_participant=_normalize_trial_cap(config.max_trials_per_class_per_participant),\n',
    '        max_trials_per_class_per_participant=_normalize_trial_cap(config.max_trials_per_class_per_participant),\n        trial_selection=_normalize_trial_selection(\n            getattr(config, "trial_selection", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION)\n        ),\n        trial_selection_seed=_normalize_trial_selection_seed(\n            getattr(config, "trial_selection_seed", DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED)\n        ),\n',
)
core = insert_after(
    core,
    'def _normalize_trial_cap(value):\n    if value is None:\n        return None\n    value = int(value)\n    if value <= 0:\n        raise ValueError("max_trials_per_class_per_participant must be positive.")\n    return value\n\n\n',
    'def _normalize_trial_selection(value):\n    normalized = str(value).strip().lower().replace("-", "_")\n    if normalized not in TRIAL_SELECTION_MODES:\n        raise ValueError(f"trial_selection must be one of {TRIAL_SELECTION_MODES}.")\n    return normalized\n\n\ndef _normalize_trial_selection_seed(value):\n    if value is None or value == "":\n        return None\n    value = int(value)\n    if value < 0:\n        raise ValueError("trial_selection_seed must be non-negative or None.")\n    return value\n\n\n',
)
write(core_path, core)

cli_path = "src/pymegdec/stimulus_cli.py"
cli = read(cli_path)
cli = replace_once(
    cli,
    '    DEFAULT_CROSS_SUBJECT_NORMALIZATION,\n    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,\n',
    '    DEFAULT_CROSS_SUBJECT_NORMALIZATION,\n    DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,\n    DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,\n    DEFAULT_CROSS_SUBJECT_PARTICIPANTS,\n',
)
cli = replace_once(
    cli,
    '    CrossSubjectStimulusConfig,\n    export_cross_subject_stimulus_smoke,\n',
    '    CrossSubjectStimulusConfig,\n    TRIAL_SELECTION_MODES,\n    export_cross_subject_stimulus_smoke,\n',
)
trial_args = '''    parser.add_argument(
        "--trial-selection",
        choices=TRIAL_SELECTION_MODES,
        default=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION,
        help="Trial subset policy used when --max-trials-per-class-per-participant is set. 'random' samples a seeded subset; 'first' keeps legacy file-order trials.",
    )
    parser.add_argument(
        "--trial-selection-seed",
        type=int,
        default=DEFAULT_CROSS_SUBJECT_TRIAL_SELECTION_SEED,
        help="Seed for random trial selection; ignored with --trial-selection first.",
    )
'''
cli = insert_after(cli, '        help="Optional deterministic cap on trials per stimulus class and participant for quick screening.",\n    )\n', trial_args)
cli = insert_after(cli, '        help="Optional deterministic cap on trials per stimulus class and participant for quick nested screening.",\n    )\n', trial_args)
cli = replace_after(cli, 'def stimulus_cross_subject_smoke', '        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,\n', '        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,\n        trial_selection=args.trial_selection,\n        trial_selection_seed=args.trial_selection_seed,\n')
cli = replace_after(cli, 'candidate_configs = make_cross_subject_candidate_configs', '        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,\n', '        max_trials_per_class_per_participant=args.max_trials_per_class_per_participant,\n        trial_selection=args.trial_selection,\n        trial_selection_seed=args.trial_selection_seed,\n')
write(cli_path, cli)

tests_path = "tests/test_stimulus_cross_subject.py"
tests = read(tests_path)
tests = replace_once(
    tests,
    '            max_trials_per_class_per_participant=2,\n            chance_classes=2,\n',
    '            max_trials_per_class_per_participant=2,\n            trial_selection="first",\n            chance_classes=2,\n',
)
tests = insert_after(
    tests,
    '        self.assertEqual(feature_set.features.shape[0], 4)\n        self.assertEqual(feature_set.max_trials_per_class_per_participant, 2)\n\n',
    '''    def test_trial_cap_random_selection_is_seeded_and_not_file_order(self):
        labels = np.asarray([1, 2, 1, 2, 1, 2], dtype=int)

        selected = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
            labels,
            2,
            selection="random",
            seed=0,
            participant=1,
        )
        repeated = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
            labels,
            2,
            selection="random",
            seed=0,
            participant=1,
        )
        legacy = cross_subject._selected_trial_indices(  # pylint: disable=protected-access
            labels,
            2,
            selection="first",
            seed=0,
            participant=1,
        )

        self.assertEqual(selected.tolist(), [1, 2, 3, 4])
        self.assertEqual(repeated.tolist(), selected.tolist())
        self.assertEqual(legacy.tolist(), [0, 1, 2, 3])

    def test_random_trial_cap_preserves_original_trial_indices(self):
        data_by_participant = {1: _mat_data([1, 2, 1, 2, 1, 2], [-1.0, 1.0, -0.9, 0.9, -0.8, 0.8])}
        config = CrossSubjectStimulusConfig(
            window_center=0.2,
            window_size=0.1,
            normalization="none",
            components_pca=float("inf"),
            max_trials_per_class_per_participant=2,
            trial_selection="random",
            trial_selection_seed=0,
            chance_classes=2,
        )

        with patch("pymegdec.stimulus_cross_subject.sio.loadmat", side_effect=_loadmat_side_effect(data_by_participant)):
            feature_set = load_participant_stimulus_features("unused", 1, config=config)

        self.assertEqual(feature_set.trial_indices.tolist(), [1, 2, 3, 4])
        self.assertEqual(feature_set.labels.tolist(), [2, 1, 2, 1])

''',
)
write(tests_path, tests)

stim_doc_path = "docs/stimulus-decoding.md"
stim_doc = read(stim_doc_path)
stim_doc = replace_once(
    stim_doc,
    'The trial cap is a deterministic screening option: it keeps the first `N`\ntrials of each stimulus class for each participant, preserving nested LOSO\nwhile making candidate selection fast enough to iterate. Omit\n`--max-trials-per-class-per-participant` for the final all-trial benchmark.\n',
    'The trial cap is a deterministic screening option: by default it draws a seeded\nrandom subset of `N` trials from each stimulus class for each participant,\npreserving nested LOSO while avoiding a file-order or block-order bias in fast\ncandidate-selection runs. Use `--trial-selection-seed` to reproduce a screening\nsubset, and use `--trial-selection first` only when you intentionally need the\nlegacy first-trials-per-class behavior. Omit `--max-trials-per-class-per-participant`\nfor the final all-trial benchmark.\n',
)
write(stim_doc_path, stim_doc)

cli_doc_path = "docs/cli.md"
cli_doc = read(cli_doc_path)
cli_doc = replace_once(
    cli_doc,
    'pymegdec stimulus cross-subject-nested --participants 1-4,6,8,9,10,13-27 --window-centers 0.150,0.175,0.200 --classifiers multinomial-logistic,shrinkage-lda,multiclass-svm --max-trials-per-class-per-participant 10\n',
    'pymegdec stimulus cross-subject-nested --participants 1-4,6,8,9,10,13-27 --window-centers 0.150,0.175,0.200 --classifiers multinomial-logistic,shrinkage-lda,multiclass-svm --max-trials-per-class-per-participant 10 --trial-selection random --trial-selection-seed 0\n',
)
write(cli_doc_path, cli_doc)

subprocess.run(["git", "diff", "--check"], check=True)
subprocess.run(["python", "-m", "compileall", "-q", "src/pymegdec", "tests/test_stimulus_cross_subject.py"], check=True)
print("trial selection fix applied")
