#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-${PYMEGDEC_DATA_DIR:-data}}"
PARTICIPANTS="${PARTICIPANTS:-1-4,6,8,9,10,13-27}"
OUT_DIR="${OUT_DIR:-outputs/alignment_loso}"
ALIGNMENT_DATA="${ALIGNMENT_DATA:-main}"
COMMON=(--data-dir "$DATA_DIR" --participants "$PARTICIPANTS" --window-center 0.175 --window-size 0.1 --alignment-data "$ALIGNMENT_DATA" --feature-mode sensor_flat --normalization subject_baseline_z --classifier multinomial-logistic --classifier-param 1.0 --components-pca 64 --chance-classes 16 --signflip-permutations 10000 --signflip-seed 0)
mkdir -p "$OUT_DIR"

pymegdec stimulus cross-subject-mcca \
  "${COMMON[@]}" \
  --mcca-components 64 \
  --mcca-regularization 1e-6 \
  --mcca-sample-mode class_repetition \
  --mcca-repetitions-per-class 10 \
  --target-centering target_unsupervised \
  --outer-output "$OUT_DIR/mcca_outer.csv" \
  --summary-output "$OUT_DIR/mcca_group_summary.csv" \
  --predictions-output "$OUT_DIR/mcca_predictions.csv" \
  --confusion-output "$OUT_DIR/mcca_confusion.csv" \
  --per-stimulus-output "$OUT_DIR/mcca_per_stimulus.csv" \
  --confusion-pairs-output "$OUT_DIR/mcca_confusion_pairs.csv"

pymegdec stimulus cross-subject-hyperalignment \
  "${COMMON[@]}" \
  --hyperalignment-components 64 \
  --hyperalignment-iterations 10 \
  --hyperalignment-sample-mode class_repetition \
  --hyperalignment-repetitions-per-class 10 \
  --target-centering target_unsupervised \
  --outer-output "$OUT_DIR/hyperalignment_outer.csv" \
  --summary-output "$OUT_DIR/hyperalignment_group_summary.csv" \
  --predictions-output "$OUT_DIR/hyperalignment_predictions.csv" \
  --confusion-output "$OUT_DIR/hyperalignment_confusion.csv" \
  --per-stimulus-output "$OUT_DIR/hyperalignment_per_stimulus.csv" \
  --confusion-pairs-output "$OUT_DIR/hyperalignment_confusion_pairs.csv"

pymegdec stimulus cross-subject-mcca \
  "${COMMON[@]}" --window-center -0.175 \
  --mcca-components 64 --mcca-regularization 1e-6 --mcca-sample-mode class_repetition --mcca-repetitions-per-class 10 \
  --outer-output "$OUT_DIR/prestim_mcca_outer.csv" \
  --summary-output "$OUT_DIR/prestim_mcca_group_summary.csv" \
  --predictions-output "$OUT_DIR/prestim_mcca_predictions.csv" \
  --confusion-output "$OUT_DIR/prestim_mcca_confusion.csv" \
  --per-stimulus-output "$OUT_DIR/prestim_mcca_per_stimulus.csv" \
  --confusion-pairs-output "$OUT_DIR/prestim_mcca_confusion_pairs.csv"

pymegdec stimulus cross-subject-hyperalignment \
  "${COMMON[@]}" --window-center -0.175 \
  --hyperalignment-components 64 --hyperalignment-iterations 10 --hyperalignment-sample-mode class_repetition --hyperalignment-repetitions-per-class 10 \
  --outer-output "$OUT_DIR/prestim_hyperalignment_outer.csv" \
  --summary-output "$OUT_DIR/prestim_hyperalignment_group_summary.csv" \
  --predictions-output "$OUT_DIR/prestim_hyperalignment_predictions.csv" \
  --confusion-output "$OUT_DIR/prestim_hyperalignment_confusion.csv" \
  --per-stimulus-output "$OUT_DIR/prestim_hyperalignment_per_stimulus.csv" \
  --confusion-pairs-output "$OUT_DIR/prestim_hyperalignment_confusion_pairs.csv"

pymegdec stimulus cross-subject-mcca \
  "${COMMON[@]}" --label-shuffle-control --label-shuffle-seed 0 \
  --mcca-components 64 --mcca-regularization 1e-6 --mcca-sample-mode class_repetition --mcca-repetitions-per-class 10 \
  --outer-output "$OUT_DIR/shuffle_mcca_outer.csv" \
  --summary-output "$OUT_DIR/shuffle_mcca_group_summary.csv" \
  --predictions-output "$OUT_DIR/shuffle_mcca_predictions.csv" \
  --confusion-output "$OUT_DIR/shuffle_mcca_confusion.csv" \
  --per-stimulus-output "$OUT_DIR/shuffle_mcca_per_stimulus.csv" \
  --confusion-pairs-output "$OUT_DIR/shuffle_mcca_confusion_pairs.csv"

pymegdec stimulus cross-subject-hyperalignment \
  "${COMMON[@]}" --label-shuffle-control --label-shuffle-seed 0 \
  --hyperalignment-components 64 --hyperalignment-iterations 10 --hyperalignment-sample-mode class_repetition --hyperalignment-repetitions-per-class 10 \
  --outer-output "$OUT_DIR/shuffle_hyperalignment_outer.csv" \
  --summary-output "$OUT_DIR/shuffle_hyperalignment_group_summary.csv" \
  --predictions-output "$OUT_DIR/shuffle_hyperalignment_predictions.csv" \
  --confusion-output "$OUT_DIR/shuffle_hyperalignment_confusion.csv" \
  --per-stimulus-output "$OUT_DIR/shuffle_hyperalignment_per_stimulus.csv" \
  --confusion-pairs-output "$OUT_DIR/shuffle_hyperalignment_confusion_pairs.csv"
