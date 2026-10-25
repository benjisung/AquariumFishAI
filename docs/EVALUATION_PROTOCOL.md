# Final test evaluation protocol

## Test integrity

- The final evaluation uses a manually annotated 60-frame set from a later full-tank recording.
- Frame selection was fixed by duration and timestamp spacing before image extraction and did not use predictions.
- Ground truth was manually audited and frozen before v1/v2 inference.
- No final-test images were used for training, fine-tuning, threshold selection, or checkpoint selection.
- The complete source frames, labels, prediction files, and row-level metrics are withheld.

## Fixed settings

- Image size: `960`
- Thresholded detection confidence: `0.35`
- Class-aware one-to-one ground-truth matching: IoU `>= 0.50`
- AP inference floor: `0.001`
- AP interpolation: 101-point precision-recall interpolation
- mAP@50:95 IoU thresholds: `0.50, 0.55, ..., 0.95`
- mAP classes: `0, 1, 4, 5, 6, 7`

Classes 2 and 3 had zero ground-truth boxes. Their AP is undefined and they are excluded from mAP. Their thresholded detections still contribute false positives.

## Paired frame-level bootstrap

- 10,000 replicates, seed `20260918`.
- Each replicate draws 60 frame IDs with replacement.
- The same sample and multiplicities are used for v1 and v2.
- Thresholded TP/FP/FN and AP are recomputed for every replicate.
- AP is recomputed after expanding duplicate frame occurrences and globally ranking all retained detections by confidence per class.
- Fixed target-class AP remains undefined if a replicate has no GT for any target class; it is never replaced by zero or computed on a reduced class set.
- Paired metric differences are computed within each replicate before percentile intervals are calculated.
- No replicate had zero GT for a fixed target class in this run.

The reported intervals quantify frame-sampling variability within this one recording. They do not capture variation among tanks, cameras, days, or annotation teams.
