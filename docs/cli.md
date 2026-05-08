# CLI reference

PyMEGDec exposes one grouped command plus compatibility entry points. Prefer the
grouped `pymegdec` command for new documentation and scripts.

## Grouped command

```bash
pymegdec --help
```

Available subcommands:

```bash
pymegdec cross-validate --participant 2
pymegdec transfer --participant 2 --classifier multiclass-svm
pymegdec stimulus-decoding --participants 2 --output outputs/part2_stimulus_decoding.csv
pymegdec make-synthetic-data --out demo-data
pymegdec alpha-movement-results --movement-summary outputs/part2_alpha_movement_summary.csv --effect-output outputs/part2_alpha_movement_effects.csv --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv
```

## Compatibility entry points

The package also installs these script names:

```bash
pymegdec-cross-validate
pymegdec-transfer
pymegdec-stimulus-decoding
pymegdec-alpha-movement-results
pymegdec-make-synthetic-data
```

Top-level Python wrappers remain available for existing workflows, for example:

```bash
python analyze_stimulus_decoding.py --participants 2 --output outputs/part2_stimulus_decoding.csv
python export_alpha_metrics.py --participant 2 --output outputs/part2_alpha_metrics.csv
python analyze_alpha_movement.py --participants 2 --trajectory-output outputs/part2_alpha_movement.csv --summary-output outputs/part2_alpha_movement_summary.csv
```

## Shared decoding options

The cross-validation and transfer commands share the core decoding options:

| Option | Meaning | Typical value |
| --- | --- | --- |
| `--data-dir` | Directory containing participant MAT files. | `/path/to/MEG-Data` |
| `--participant` | Participant id. | `2` |
| `--window-size` | Window duration in seconds. | `0.1` |
| `--train-window-center` | Stimulus training-window center in seconds. | `0.2` |
| `--null-window-center` | Null-window center, or `nan`. | `nan` or `-0.2` |
| `--new-framerate` | Target frame rate, or `inf`. | `inf` |
| `--classifier` | Classifier registry name. | `multiclass-svm` |
| `--classifier-param` | Numeric, JSON, or Python literal parameter. | `1.0` |
| `--components-pca` | PCA component count, or `inf`. | `100` |
| `--frequency-range LOW HIGH` | Frequency range in Hz. | `0 inf` |

## Synthetic demo data

Generate balanced private-data-free MATLAB files for participant 2:

```bash
pymegdec make-synthetic-data --out demo-data
```

The generator writes `Part2Data.mat`, `Part2CueData.mat`, and
`synthetic_data_manifest.json`. The generated files follow the same participant
file naming convention as the private MEG data and can be used with the standard
commands:

```bash
pymegdec cross-validate --data-dir demo-data --participant 2
pymegdec transfer --data-dir demo-data --participant 2 --null-window-center nan
```

Customize the generated dataset when testing edge cases:

```bash
pymegdec make-synthetic-data \
  --out demo-data \
  --participant 3 \
  --classes 4 \
  --main-repeats 8 \
  --cue-repeats 4 \
  --channels 6 \
  --overwrite
```

## Cross-validation

Cross-validate one participant's main dataset:

```bash
pymegdec cross-validate --data-dir /path/to/MEG-Data --participant 2 --folds 10
```

The command prints the accuracy returned by
`pymegdec.cross_validation.cross_validate_single_dataset`.

## Model transfer

Train on the main experiment file and validate on the cue file for one
participant:

```bash
pymegdec transfer --data-dir /path/to/MEG-Data --participant 2 --null-window-center nan
```

The command prints the accuracy returned by
`pymegdec.model_transfer.evaluate_model_transfer`.

## Stimulus decoding

Run train-main / validate-cue decoding across a time range:

```bash
pymegdec stimulus-decoding \
  --data-dir /path/to/MEG-Data \
  --participants 2 \
  --time-window=-0.2,0.6 \
  --window-step-s 0.05 \
  --output outputs/part2_stimulus_decoding.csv \
  --summary-output outputs/part2_stimulus_decoding_summary.csv \
  --plots-dir outputs/part2_stimulus_decoding_plots
```

Use `--transfer-direction cue-to-main` to train on cue data and validate on the
main experiment data.

## Alpha movement result analysis

Analyze a movement summary exported by `analyze_alpha_movement.py`:

```bash
pymegdec alpha-movement-results \
  --movement-summary outputs/part2_alpha_movement_summary.csv \
  --effect-output outputs/part2_alpha_movement_effects.csv \
  --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv \
  --plots-dir outputs/part2_alpha_movement_plots
```
