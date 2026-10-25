# Dataset card

## Summary

The project uses three related collections: a controlled-box dataset for initial training, a small full-tank adaptation set, and a later independent full-tank test. The original images and YOLO annotation files are not included here.

## Class set

Eight numeric categories are listed in `../configs/classes.txt`. Classes 2 and 3 were absent from final-test ground truth. Class 7 has a historical naming discrepancy: legacy internal model files say `Platax_sp`, while the manuscript-facing final identity is *Monodactylus argenteus* (Silver moony).

## Development data and split design

### Controlled v1

- 419 images, extracted every two seconds from one controlled-box source recording.
- First 346 reviewed images were shuffled with seed 42 and split into 276 training and 70 validation images.
- The later 73 auto-labeled images were assigned to training only.
- Final on-disk counts: 349 train and 70 validation.
- Because the random split is at image level within one continuous video, adjacent temporal frames may occur in both sets. This validation is development evidence and can overestimate performance on new footage.

### Full-tank v2 adaptation

- 60 manually labeled images sampled at one frame per second from a 60-second full-tank recording.
- Seeded image-level split: 48 adaptation training and 12 development validation images.
- The 419 controlled-origin images, including the earlier v1 validation frames, were all used as part of v2 training.
- Therefore v2 development validation is from the adaptation source session and is not independent testing.

## Frozen final test

- Source: a separate later full-tank recording, duration 6,033.4 seconds.
- Selection: 60 deterministic timestamps approximately evenly spaced over the usable recording duration, with neutral 5-second start/end margins.
- Mean consecutive-frame spacing: 102.0915 seconds; range 102.091–102.092 seconds.
- Manual human annotations: 428 boxes.
- The final frames were excluded from both training datasets and postdate the evaluated checkpoints. No byte-identical image overlap was found against either training dataset.
- Raw frames, labels, timestamps, and row-level evaluation files are not distributed.

## Annotation and exclusions

Images were manually annotated in YOLO bounding-box format. Adaptation labels left equipment and unclear objects unlabeled as background. This provides hard-negative context but may also reflect project-specific annotation decisions. Empty or absent classes must not be interpreted as evidence of biological absence from the complete aquarium.

## Data access and privacy

The complete source videos and images are withheld. This repository publishes aggregate metrics, per-class metrics, methodology, and selected derived figures. Access to the source data requires separate permission from the data owner.

## Recommended interpretation

The final test measures performance on new footage from the same aquarium deployment domain. It does not measure transfer to independent tanks, cameras, or wild reefs. Since all test frames come from one recording, the test does not represent between-tank or between-site variability.
