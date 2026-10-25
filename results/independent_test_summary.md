# CJSJ Final Validation 60: v1 vs v2

## Final test design

- 60 manually annotated, prediction-independent frames from the final full-tank recording.
- Ground truth was audited and frozen before either model was evaluated.
- Identical settings: `imgsz=960`, confidence `0.35`, class-aware matching IoU `0.5`, device `0`.
- AP uses the verified prior evaluator's 101-point interpolation with inference floor `0.001`.
- mAP averages include only positive-GT classes 0, 1, 4, 5, 6, and 7.
- Classes 2 and 3 are excluded from AP averages; their thresholded predictions still count as false positives.

## Frozen GT hash verification

- Before inference: 60/60 images and 60/60 labels matched the freeze manifest.
- After evaluation: 60/60 images and 60/60 labels matched the freeze manifest.

## Aggregate results

| Model | TP | FP | FN | Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 29 | 280 | 399 | 0.093851 | 0.067757 | 0.078697 | 0.073331 | 0.042952 |
| v2 | 226 | 79 | 202 | 0.740984 | 0.528037 | 0.616644 | 0.582021 | 0.303403 |

## False-positive change

- Signed change from v1 to v2: -71.79%
- False-positive reduction from v1 to v2: 71.79%

## Per-class results

| Model | ID | Manuscript name | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50:95 | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 | 0 | Paracanthurus_hepatus | 91 | 21 | 11 | 70 | 0.65625000 | 0.23076923 | 0.34146341 | 0.32125432 | 0.20060398 | present |
| v1 | 1 | Nemateleotris_magnifica | 101 | 0 | 12 | 101 | 0.00000000 | 0.00000000 | 0.00000000 | 0.01064979 | 0.00429773 | present |
| v1 | 2 | Amphiprion_ocellaris_orange | 0 | 0 | 19 | 0 | 0.00000000 | N/A | N/A | N/A | N/A | absent_from_GT |
| v1 | 3 | Amphiprion_ocellaris_black | 0 | 0 | 144 | 0 | 0.00000000 | N/A | N/A | N/A | N/A | absent_from_GT |
| v1 | 4 | Dascyllus_aruanus | 58 | 8 | 65 | 50 | 0.10958904 | 0.13793103 | 0.12213740 | 0.06269095 | 0.02480020 | present |
| v1 | 5 | Pomacentrus_moluccensis | 31 | 0 | 0 | 31 | N/A | 0.00000000 | N/A | 0.03379013 | 0.02165687 | present |
| v1 | 6 | Chromis_viridis | 103 | 0 | 24 | 103 | 0.00000000 | 0.00000000 | 0.00000000 | 0.01102999 | 0.00618984 | present |
| v1 | 7 | Monodactylus_argenteus | 44 | 0 | 5 | 44 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00057121 | 0.00016596 | present |
| v2 | 0 | Paracanthurus_hepatus | 91 | 51 | 8 | 40 | 0.86440678 | 0.56043956 | 0.68000000 | 0.65489284 | 0.35512692 | present |
| v2 | 1 | Nemateleotris_magnifica | 101 | 43 | 17 | 58 | 0.71666667 | 0.42574257 | 0.53416149 | 0.56049871 | 0.25961911 | present |
| v2 | 2 | Amphiprion_ocellaris_orange | 0 | 0 | 4 | 0 | 0.00000000 | N/A | N/A | N/A | N/A | absent_from_GT |
| v2 | 3 | Amphiprion_ocellaris_black | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | absent_from_GT |
| v2 | 4 | Dascyllus_aruanus | 58 | 5 | 2 | 53 | 0.71428571 | 0.08620690 | 0.15384615 | 0.25123972 | 0.09119709 | present |
| v2 | 5 | Pomacentrus_moluccensis | 31 | 0 | 0 | 31 | N/A | 0.00000000 | N/A | 0.36957105 | 0.19462591 | present |
| v2 | 6 | Chromis_viridis | 103 | 91 | 29 | 12 | 0.75833333 | 0.88349515 | 0.81614350 | 0.84197744 | 0.45649692 | present |
| v2 | 7 | Monodactylus_argenteus | 44 | 36 | 19 | 8 | 0.65454545 | 0.81818182 | 0.72727273 | 0.81394371 | 0.46335296 | present |

## Absent-class false positives

- v1, class 2 `Amphiprion_ocellaris_orange`: 19 FP
- v1, class 3 `Amphiprion_ocellaris_black`: 144 FP
- v2, class 2 `Amphiprion_ocellaris_orange`: 4 FP
- v2, class 3 `Amphiprion_ocellaris_black`: 0 FP

## Limitations

- The test contains 60 frames from one final-condition recording, so uncertainty and scene diversity remain limited.
- Classes 2 and 3 have no positive GT instances; sensitivity and AP cannot be estimated for them.
- AP is calculated by the project's verified custom evaluator and is conditional on predictions retained at the 0.001 inference floor.
- Class 7 is matched by numeric ID. The manuscript name is `Monodactylus_argenteus`; historical model metadata may say `Platax_sp`.
- Frames are sampled from a video and may not be statistically independent in the same way as recordings from different tanks or days.
