# Development validation artifacts

Plots under `results/development_validation/v1/` and `v2/` were copied from the respective original YOLO training runs. They include epoch-level training results, precision-recall curves, and normalized/unnormalized confusion matrices from each run's configured validation split.

## v1

The validation split contains 70 images randomly sampled at image level from the same controlled-box recording as training. Adjacent two-second frames may cross the split. Treat its curves as same-recording development diagnostics, not an independent generalization estimate.

## v2

The v2 training split includes 419 controlled-origin images plus 48 full-tank adaptation images. The validation split contains 12 images from the same 60-second full-tank adaptation recording. The controlled-origin images previously used for v1 validation are part of v2 training. Treat these 12 images as target-domain development validation, not final independent test evidence.

The main README reports final comparison metrics from a later frozen 60-frame test and includes its separate confusion matrices under `results/figures/`.
