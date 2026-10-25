# Corrected Paired Frame-Level Bootstrap 95% Confidence Intervals

## Fixed method

- Frozen frames per replicate: 60, sampled with replacement.
- Replicates: 10,000
- Random seed: 20260918
- Identical ordered frame samples were used for v1 and v2.
- Thresholded metrics: confidence 0.35, class-aware one-to-one matching, IoU 0.5.
- AP: literal occurrence expansion, confidence floor 0.001, verified 101-point interpolation, IoU 0.50:0.95.
- Fixed mAP classes: 0, 1, 4, 5, 6, 7.
- Percentile interval: 2.5th and 97.5th percentiles using NumPy linear interpolation.
- No model inference or training was run.

## Expanded-AP validation

- Result: **PASS**
- Maximum absolute difference from literal verified evaluation: `0`

## Corrected bootstrap results

| Comparison | Metric | Point estimate | Bootstrap mean | 2.5% | 97.5% | Valid | Undefined |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | TP | 29.0000000000 | 28.9322000000 | 21.0000000000 | 37.0000000000 | 10000 | 0 |
| v1 | FP | 280.0000000000 | 280.0692000000 | 258.0000000000 | 302.0000000000 | 10000 | 0 |
| v1 | FN | 399.0000000000 | 398.9268000000 | 377.0000000000 | 421.0000000000 | 10000 | 0 |
| v1 | Precision | 0.0938511300 | 0.0936322773 | 0.0684931507 | 0.1187500000 | 10000 | 0 |
| v1 | Recall | 0.0677570100 | 0.0676088077 | 0.0494382022 | 0.0859950860 | 10000 | 0 |
| v1 | F1 | 0.0786974200 | 0.0784869588 | 0.0576131687 | 0.0993288591 | 10000 | 0 |
| v2 | TP | 226.0000000000 | 225.9907000000 | 204.0000000000 | 248.0000000000 | 10000 | 0 |
| v2 | FP | 79.0000000000 | 78.9182000000 | 63.0000000000 | 96.0000000000 | 10000 | 0 |
| v2 | FN | 202.0000000000 | 201.8683000000 | 181.0000000000 | 223.0000000000 | 10000 | 0 |
| v2 | Precision | 0.7409836100 | 0.7412354890 | 0.6895519949 | 0.7900000000 | 10000 | 0 |
| v2 | Recall | 0.5280373800 | 0.5281902692 | 0.4847040745 | 0.5711141567 | 10000 | 0 |
| v2 | F1 | 0.6166439300 | 0.6165910317 | 0.5751435784 | 0.6560010624 | 10000 | 0 |
| v1 | mAP50 | 0.0733310700 | 0.0780557212 | 0.0575602011 | 0.1044803981 | 10000 | 0 |
| v1 | mAP50_95 | 0.0429524300 | 0.0462429995 | 0.0327789729 | 0.0633174073 | 10000 | 0 |
| v2 | mAP50 | 0.5820205800 | 0.5888370538 | 0.5375210787 | 0.6412567946 | 10000 | 0 |
| v2 | mAP50_95 | 0.3034031500 | 0.3098534685 | 0.2843737439 | 0.3357831044 | 10000 | 0 |
| v2_minus_v1 | Precision | 0.6471324800 | 0.6476032118 | 0.5958175392 | 0.6979849301 | 10000 | 0 |
| v2_minus_v1 | Recall | 0.4602803700 | 0.4605814615 | 0.4199523292 | 0.5025452698 | 10000 | 0 |
| v2_minus_v1 | F1 | 0.5379465100 | 0.5381040729 | 0.4985928371 | 0.5766191615 | 10000 | 0 |
| v2_minus_v1 | mAP50 | 0.5086895100 | 0.5107813326 | 0.4621078821 | 0.5610718199 | 10000 | 0 |
| v2_minus_v1 | mAP50_95 | 0.2604507200 | 0.2636104690 | 0.2385137770 | 0.2893824350 | 10000 | 0 |
| v2_minus_v1 | FP | -201.0000000000 | -201.1510000000 | -229.0000000000 | -173.0000000000 | 10000 | 0 |
| v1_minus_v2 | FP_absolute_reduction | 201.0000000000 | 201.1510000000 | 173.0000000000 | 229.0000000000 | 10000 | 0 |
| v1_to_v2 | FP_percent_reduction | 71.7857142857 | 71.7682363367 | 64.9441675387 | 77.9661016949 | 10000 | 0 |

## Zero-GT fixed-class replicates

| Class ID | Class name | Undefined replicates |
|---:|---|---:|
| 0 | Paracanthurus_hepatus | 0 |
| 1 | Nemateleotris_magnifica | 0 |
| 4 | Dascyllus_aruanus | 0 |
| 5 | Pomacentrus_moluccensis | 0 |
| 6 | Chromis_viridis | 0 |
| 7 | Monodactylus_argenteus | 0 |

Replicates with any fixed target class absent: **0**.
Such replicates remain undefined; AP is not set to zero and the class set is not reduced.

## Cross-checks

- Thresholded Precision/Recall/F1 summaries unchanged: **PASS**
- Protected GT, images, predictions, weights, old scripts, and old results unchanged: **PASS**
- Google Docs were not accessed or updated.
