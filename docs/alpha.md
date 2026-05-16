# Alpha analyses

PyMEGDec contains exploratory alpha-band workflows for per-trial alpha metrics,
sensor-level alpha topography movement, and alpha/reaction-time associations.

## Alpha metrics

Prestimulus alpha metrics are exported per trial for downstream plotting or
statistics. The default channel selector uses occipital CTF channels matching
`MLO*`, `MRO*`, and `MZO*`. The default time window is `-0.4` to `-0.05 s`
before stimulus onset, and the default alpha band is `8–12 Hz`.

```powershell
python export_alpha_metrics.py --participant 2 --output outputs\part2_alpha_metrics.csv
python export_alpha_metrics.py --participant 2 --cue --output outputs\part2_cue_alpha_metrics.csv
```

The exported rows include:

- Alpha power.
- Phase concentration.
- Planar phase-fit quality.
- Spatial phase frequency.
- Estimated propagation speed.
- Dominant phase-gradient direction on a projected sensor plane. The projected
  plane is PCA/SVD-fitted, but its axes are anchored to the original sensor
  coordinate frame to avoid arbitrary SVD sign or rotation flips.

The `outputs/` directory is ignored by git.

## Sensor-level alpha movement

The MAT files contain CTF sensor geometry in `data.grad.chanpos`, with positions
in millimeters. PyMEGDec uses this geometry for sensor-array analyses of alpha
topography. The movement exporter does not infer source-localized anatomical
motion.

For each trial and sampled time point, the exporter filters selected MEG
channels to the alpha band, computes alpha power, and writes the power-weighted
centroid over the MEG sensor positions.

Defaults:

- Channel pattern: `^M`, covering all MEG channels.
- Alpha band: `8–12 Hz`.
- Time window: `-0.4` to `0.8 s` around stimulus onset.

```powershell
python analyze_alpha_movement.py `
  --participants 2 `
  --trajectory-output outputs\part2_alpha_movement.csv `
  --summary-output outputs\part2_alpha_movement_summary.csv
```

The trajectory CSV includes 3D CTF sensor centroids, projected 2D centroids,
stepwise speed, displacement from the first sampled time point, the peak-power
channel, and a spatial concentration score. Projected centroids use the same
coordinate-anchored sensor plane as the alpha phase-gradient metrics: projected
x follows global +x when possible, projected y follows global +y when possible,
and axes that are normal to the fitted plane are skipped. Treat the trajectory as
movement of the measured alpha topography over sensors, not as anatomical source
motion.

## Alpha movement result analysis

Analyze movement summaries into pre/post-stimulus effects and condition-level
plots:

```powershell
python analyze_alpha_movement_results.py `
  --movement-summary outputs\part2_alpha_movement_summary.csv `
  --effect-output outputs\part2_alpha_movement_effects.csv `
  --condition-summary-output outputs\part2_alpha_movement_condition_summary.csv `
  --plots-dir outputs\part2_alpha_movement_plots
```

Equivalent grouped command:

```bash
pymegdec alpha-movement-results \
  --movement-summary outputs/part2_alpha_movement_summary.csv \
  --effect-output outputs/part2_alpha_movement_effects.csv \
  --condition-summary-output outputs/part2_alpha_movement_condition_summary.csv \
  --plots-dir outputs/part2_alpha_movement_plots
```

The effects compare the mean pre-stimulus centroid with the mean post-stimulus
centroid and summarize speed, alpha power, and spatial concentration changes.

## Alpha and reaction time

Saved `Part*Data.mat` files may not contain reaction times. The RT analysis
command therefore accepts an external behavioral CSV with these columns:

```text
participant,trial,reaction_time
```

The external CSV `trial` column is interpreted as a zero-based trial index by
default, matching the alpha-metrics export. For common behavioral files numbered
`1..N`, pass `--reaction-time-trial-base 1`; PyMEGDec converts those trial
numbers to zero-based indices before joining. A likely unconverted one-based CSV
raises an error instead of silently shifting alpha/RT pairs by one trial.

If reaction times are stored in a future MAT `trialinfo` column, pass
`--trialinfo-rt-column` instead.

```powershell
python analyze_alpha_reaction_time.py `
  --participants 2 `
  --reaction-times behavior_rt.csv `
  --joined-output outputs\part2_alpha_rt_trials.csv `
  --summary-output outputs\part2_alpha_rt_summary.csv
```

The summary includes per-participant Pearson/regression rows and a pooled
within-participant row for each alpha metric. Phase-gradient direction is encoded
as sine and cosine components before analysis.
