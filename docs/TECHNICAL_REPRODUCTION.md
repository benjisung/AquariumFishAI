# Technical reproduction notes

## Inference demonstrations

The two demonstration scripts use command-line source paths and the bundled v2 checkpoint by default:

- `scripts/render_demo_video.py`
- `scripts/analyze_tracking_video.py`

The sample commands in the main README are intended for user-owned or otherwise authorized source videos. They write generated videos, tables, and figures beneath `outputs/`, which is excluded from version control by `.gitignore`.

## Reproducing the final comparison

The frozen test images and labels are withheld. With separately authorized access to the exact frozen files and their original freeze manifest, place them in a private directory (or provide paths explicitly) and run:

```powershell
python scripts\evaluation\compare_final_test.py --images path\to\private_final_test\images --labels path\to\private_final_test\labels --manifest path\to\private_final_test_freeze_manifest.csv --v1-model weights\reef_ai_yolov12n_v1_controlled.pt --v2-model weights\reef_ai_yolov12n_v2_full_tank.pt --output outputs\evaluation
```

The original image and label bytes must match the manifest; the evaluator checks hashes before and after inference. The manifest and private test files are not included in this repository.

## Reproducing the paired bootstrap

The bootstrap uses the saved raw prediction records produced by the comparison script. With authorized data and prediction records available at the evaluator's expected `outputs/evaluation/` layout, run:

```powershell
python scripts\evaluation\bootstrap_expanded_ap.py
```

This runs 10,000 paired frame-level replicates with seed `20260918`, confidence threshold `0.35`, matching IoU `0.50`, and AP confidence floor `0.001`. It does not load models or perform inference. Results already generated for this study are included under `results/`.

## Data access

Contact the maintainer for permission and access to the frozen test data. Do not substitute model predictions for human annotations or use the final test for training, fine-tuning, threshold selection, or model selection.
