# Model card: Reef AI YOLOv12n v2 full-tank detector

## Model

- File: `weights/reef_ai_yolov12n_v2_full_tank.pt`
- Architecture family: YOLOv12n
- Development: initialized from the v1 controlled-box checkpoint and fine-tuned for 60 epochs on the mixed v2 dataset.
- Input size used in final evaluation: 960 pixels.
- Evaluation confidence threshold: 0.35.
- Inference output retained to confidence 0.001 for AP calculations.
- Public-copy SHA-256: `bfe58b9d8e8b1c1091ed148a2f80b384111f823d76629bb3fce85dea2addeb57`.
- The public copy has normalized path metadata. Model and EMA tensor-state digests were checked unchanged after metadata cleanup.

## Intended use

Research demonstration and offline monitoring of fish in the project’s fixed full-tank aquarium view. It is intended to support evaluation of detection and video-analysis workflows. It is not a general species-identification system for wild reefs or arbitrary aquaria.

## Training data summary

The mixed v2 training set contains 419 controlled-origin training images and 48 manually labeled full-tank adaptation images. A separate 12-image subset from the same 60-second full-tank adaptation recording was used as development validation. Source frames and labels are withheld from this repository.

## Final independent test

The v2 checkpoint was evaluated on 60 manually annotated frames sampled from a later 6,033.4-second full-tank recording, with 428 GT boxes. The frozen test source postdates the checkpoint. Aggregate results at the fixed threshold were precision 0.7410, recall 0.5280, F1 0.6166, mAP@50 0.5820, and mAP@50:95 0.3034. Paired frame-bootstrap 95% intervals appear in `results/paired_bootstrap_95ci_summary.md`.

## Class mapping

Eight numeric classes are used. Historical internal metadata calls class 7 `Platax_sp`; manuscript-facing identification confirmed for the final test is *Monodactylus argenteus* (Silver moony). Numeric ID 7 is retained for compatibility. See `SPECIES_MAPPING.md`.

## Known limitations and risks

- The evaluation covers one tank, one general view, and one later recording.
- Classes 2 and 3 had no positive boxes in the final test; sensitivity and AP are unknown for them.
- Class 5 had 31 GT boxes, but the v2 thresholded evaluation produced zero matched detections; see per-class results.
- Performance may change under different lighting, water clarity, camera placement, tank equipment, fish size, occlusion, or species composition.
- A detected box does not establish species certainty or a unique biological individual.
- Tracking IDs may fragment or switch between fish.

## License and redistribution

This repository currently has no reuse license. The checkpoint was produced with the YOLOv12/Ultralytics ecosystem. Confirm applicable upstream licensing and obtain maintainer permission before redistributing or using the checkpoint commercially.
