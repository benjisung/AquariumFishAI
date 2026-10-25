"""Corrected paired frame bootstrap with literal AP event expansion.

This script reads only frozen ground truth and already-saved raw predictions.
It does not load either YOLO model and does not perform inference or training.

The thresholded bootstrap is intentionally identical to the locked prior
implementation.  The AP bootstrap is corrected: every sampled frame
occurrence contributes a separate copy of every prediction event, so the
precision-recall curve contains every intermediate cumulative state.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bootstrap_cjsj_final_validation_60 as prior  # noqa: E402


verified = prior.verified
EVALUATION_DIR = prior.EVALUATION_DIR
TARGET_CLASSES = prior.TARGET_CLASSES
IOU_THRESHOLDS = prior.IOU_THRESHOLDS
CLASS_NAMES = prior.CLASS_NAMES

OUTPUT_CSV = EVALUATION_DIR / "bootstrap_95ci_corrected.csv"
SUMMARY_MD = EVALUATION_DIR / "bootstrap_95ci_corrected_summary.md"
VALIDATION_MD = EVALUATION_DIR / "expanded_ap_validation.md"
COMPARISON_MD = EVALUATION_DIR / "bootstrap_correction_comparison.md"
OLD_BOOTSTRAP_CSV = EVALUATION_DIR / "bootstrap_95ci.csv"

MODEL_PATHS = (
    ROOT / "weights" / "reef_ai_yolov12n_v1_controlled.pt",
    ROOT / "weights" / "reef_ai_yolov12n_v2_full_tank.pt",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected paired 60-frame bootstrap with literal expanded AP events"
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_918)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--ap-conf", type=float, default=0.001)
    parser.add_argument(
        "--progress-every", type=int, default=500,
        help="Print AP progress after this many replicates for each model/class.",
    )
    return parser.parse_args()


def generate_samples(replicates: int, seed: int) -> np.ndarray:
    """Return the ordered 60 frame draws for every paired replicate."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, size=(replicates, 60), dtype=np.int16)


def samples_to_multiplicities(samples: np.ndarray) -> np.ndarray:
    """Keep an additive-count representation for thresholded metrics only."""
    multiplicities = np.zeros((len(samples), 60), dtype=np.int16)
    for replicate_index, sample in enumerate(samples):
        multiplicities[replicate_index] = np.bincount(sample, minlength=60)
    return multiplicities


def prepare_occurrence_templates(records: list[dict], class_id: int) -> list[dict]:
    """Precompute deterministic within-frame matching results.

    Matching never crosses image boundaries.  Therefore each sampled occurrence
    can safely receive a fresh copy of these deterministic per-frame results.
    The copied events are explicitly expanded later; no numerical event weight
    is used in AP calculation.
    """
    templates = []
    for record in records:
        class_gt = [item for item in record["gt"] if item["class_id"] == class_id]
        class_predictions = [
            item for item in record["predictions"] if item["class_id"] == class_id
        ]
        flags_by_threshold = []
        for threshold in IOU_THRESHOLDS:
            flags, _matched = verified.class_aware_match(
                class_predictions, class_gt, float(threshold)
            )
            flags_by_threshold.append(flags)
        if class_predictions:
            scores = np.asarray(
                [item["score"] for item in class_predictions], dtype=np.float64
            )
            tp_flags = np.asarray(flags_by_threshold, dtype=np.int8).T
        else:
            scores = np.empty(0, dtype=np.float64)
            tp_flags = np.zeros((0, len(IOU_THRESHOLDS)), dtype=np.int8)
        templates.append({
            "gt_count": len(class_gt),
            "scores": scores,
            "tp_flags": tp_flags,
        })
    return templates


def expanded_ap_for_sample(sample: np.ndarray, templates: list[dict]) -> np.ndarray:
    """Compute all ten AP values from literal prediction-event copies.

    The sample order defines occurrence order.  Each occurrence is copied into
    new score and TP arrays.  Stable confidence sorting then has the same tie
    order as the verified evaluator applied to a literally duplicated record
    list.  Every copied prediction creates its own cumulative PR-curve step.
    """
    total_gt = sum(templates[int(frame_index)]["gt_count"] for frame_index in sample)
    if total_gt == 0:
        return np.full(len(IOU_THRESHOLDS), np.nan, dtype=np.float64)

    event_count = sum(
        len(templates[int(frame_index)]["scores"]) for frame_index in sample
    )
    if event_count == 0:
        return np.zeros(len(IOU_THRESHOLDS), dtype=np.float64)

    # These are occurrence-specific copies.  Repeated source frames occupy
    # different slices and create separate PR-curve events.
    scores = np.empty(event_count, dtype=np.float64)
    tp_flags = np.empty((event_count, len(IOU_THRESHOLDS)), dtype=np.int8)
    offset = 0
    for frame_index in sample:
        template = templates[int(frame_index)]
        count = len(template["scores"])
        if count:
            stop = offset + count
            scores[offset:stop] = template["scores"]
            tp_flags[offset:stop] = template["tp_flags"]
            offset = stop

    # Python's verified evaluator uses a stable score-only sort.  Stable NumPy
    # sorting reproduces its occurrence-order and local-prediction-order ties.
    ranking = np.argsort(-scores, kind="stable")
    ranked_tp = tp_flags[ranking]
    cumulative_predictions = np.arange(1, event_count + 1, dtype=np.int32)
    recall_levels = np.linspace(0.0, 1.0, 101)

    ap_values = np.empty(len(IOU_THRESHOLDS), dtype=np.float64)
    for threshold_index in range(len(IOU_THRESHOLDS)):
        cumulative_tp = np.cumsum(ranked_tp[:, threshold_index], dtype=np.int32)
        precision = cumulative_tp / cumulative_predictions
        recall = cumulative_tp / total_gt
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        # Search the floating recall values themselves.  This exactly mirrors
        # verified.average_precision's `recall >= level` comparisons, including
        # floating-point boundary behavior at the 101 recall levels.
        indices = np.searchsorted(recall, recall_levels, side="left")
        values = np.zeros(101, dtype=np.float64)
        available = indices < event_count
        values[available] = envelope[indices[available]]
        ap_values[threshold_index] = float(np.mean(values))
    return ap_values


def expanded_bootstrap_ap(
    samples: np.ndarray,
    templates: list[dict],
    model_name: str,
    class_id: int,
    progress_every: int,
) -> np.ndarray:
    output = np.empty((len(samples), len(IOU_THRESHOLDS)), dtype=np.float64)
    started = time.perf_counter()
    for replicate_index, sample in enumerate(samples):
        output[replicate_index] = expanded_ap_for_sample(sample, templates)
        completed = replicate_index + 1
        if progress_every > 0 and completed % progress_every == 0:
            elapsed = time.perf_counter() - started
            rate = completed / elapsed if elapsed else 0.0
            print(
                f"Expanded AP {model_name} class {class_id}: "
                f"{completed}/{len(samples)} replicates ({rate:.1f} replicates/s)",
                flush=True,
            )
    return output


def independent_record_copy(record: dict) -> dict:
    """Create a reference-evaluator record with no shared GT/prediction objects."""
    return {
        "frame_id": str(record["frame_id"]),
        "gt": copy.deepcopy(record["gt"]),
        "predictions": copy.deepcopy(record["predictions"]),
    }


def validation_samples() -> list[tuple[str, np.ndarray]]:
    """Create deterministic duplicate patterns plus realistic random samples."""
    cases: list[tuple[str, np.ndarray]] = []
    unique = np.arange(60, dtype=np.int16)
    cases.append(("all 60 frames once", unique.copy()))

    repeated_twice = unique.copy()
    repeated_twice[-1] = 0
    cases.append(("frame 0 repeated 2x", repeated_twice))

    repeated_three_times = unique.copy()
    repeated_three_times[-2:] = 0
    cases.append(("frame 0 repeated 3x", repeated_three_times))

    multiple_repeats = unique.copy()
    multiple_repeats[-6:] = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int16)
    cases.append(("frames 0, 1, and 2 repeated multiple times", multiple_repeats))

    random_samples = generate_samples(8, 20_260_918)
    for index, sample in enumerate(random_samples):
        cases.append((f"seeded realistic random sample {index}", sample.copy()))
    return cases


def validate_expanded_ap(records_by_model: dict[str, list[dict]]) -> tuple[str, float]:
    """Compare the new calculation to literal records in the verified evaluator."""
    cases = validation_samples()
    maximum_difference = 0.0
    comparison_count = 0
    case_rows = []

    for description, sample in cases:
        duplicate_count = 60 - len(np.unique(sample))
        case_maximum = 0.0
        for model_name in ("v1", "v2"):
            literal_records = [
                independent_record_copy(records_by_model[model_name][int(frame_index)])
                for frame_index in sample
            ]
            for class_id in TARGET_CLASSES:
                templates = prepare_occurrence_templates(
                    records_by_model[model_name], class_id
                )
                calculated = expanded_ap_for_sample(sample, templates)
                reference = np.asarray([
                    verified.average_precision(
                        literal_records, class_id, float(threshold)
                    )
                    for threshold in IOU_THRESHOLDS
                ], dtype=np.float64)
                difference = np.abs(calculated - reference)
                finite = difference[np.isfinite(difference)]
                local_maximum = float(np.max(finite)) if len(finite) else 0.0
                case_maximum = max(case_maximum, local_maximum)
                maximum_difference = max(maximum_difference, local_maximum)
                comparison_count += len(IOU_THRESHOLDS)
                if not np.allclose(
                    calculated, reference, rtol=0.0, atol=1e-12, equal_nan=True
                ):
                    raise RuntimeError(
                        f"Expanded AP validation failed for {description}, "
                        f"{model_name}, class {class_id}: max difference {local_maximum}"
                    )
        case_rows.append((description, duplicate_count, case_maximum))

    lines = [
        "# Expanded AP Validation",
        "",
        "**Result: PASS**",
        "",
        "The corrected implementation was compared with literal, deep-copied frame",
        "occurrences evaluated by the project's verified 101-point AP function.",
        "Each case used both models, all fixed target classes, and IoU thresholds",
        "0.50 through 0.95.",
        "",
        f"- Deterministic samples tested: {len(cases)}",
        f"- AP values compared: {comparison_count}",
        f"- Required absolute tolerance: < 1e-12",
        f"- Maximum absolute difference: {maximum_difference:.16g}",
        "- The prior known failing random replicate (seeded sample index 3) is included.",
        "",
        "| Sample | Duplicate draws beyond unique frames | Maximum absolute difference |",
        "|---|---:|---:|",
    ]
    for description, duplicate_count, case_maximum in case_rows:
        lines.append(f"| {description} | {duplicate_count} | {case_maximum:.16g} |")
    lines.extend([
        "",
        "No model inference was run. Validation used only frozen GT and saved raw predictions.",
        "",
    ])
    return "\n".join(lines), maximum_difference


def read_old_bootstrap_rows() -> dict[tuple[str, str], dict]:
    with OLD_BOOTSTRAP_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["comparison"], row["metric"]): row for row in rows}


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    mean, lower, upper, valid, undefined = prior.summarize_distribution(values)
    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "valid": valid,
        "undefined": undefined,
    }


def format_number(value: float | None) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.10f}"


def build_comparison_markdown(
    distributions: dict,
    originals: dict,
) -> tuple[str, bool]:
    old_rows = read_old_bootstrap_rows()
    targets = [
        ("v1", "mAP50", "v1 mAP@50"),
        ("v1", "mAP50_95", "v1 mAP@50:95"),
        ("v2", "mAP50", "v2 mAP@50"),
        ("v2", "mAP50_95", "v2 mAP@50:95"),
        ("v2_minus_v1", "mAP50", "Paired v2-v1 mAP@50"),
        ("v2_minus_v1", "mAP50_95", "Paired v2-v1 mAP@50:95"),
    ]
    lines = [
        "# Bootstrap Correction Comparison",
        "",
        "The old AP bootstrap used weighted event jumps. The corrected bootstrap",
        "literally expands every sampled prediction occurrence before constructing",
        "the precision-recall curve. Point estimates are unchanged.",
        "",
        "| Metric | Statistic | Old | Corrected | Absolute change |",
        "|---|---|---:|---:|---:|",
    ]
    for comparison, metric, label in targets:
        values = next(
            value for key, value in distributions.items()
            if key[0] == comparison and key[1] == metric
        )
        corrected = distribution_summary(values)
        old = old_rows[(comparison, metric)]
        pairs = (
            ("Point estimate", float(old["original_point_estimate"]), originals[(comparison, metric)]),
            ("Bootstrap mean", float(old["bootstrap_mean"]), corrected["mean"]),
            ("Lower 95%", float(old["ci_2_5_percentile"]), corrected["lower"]),
            ("Upper 95%", float(old["ci_97_5_percentile"]), corrected["upper"]),
        )
        for statistic, old_value, corrected_value in pairs:
            absolute_change = abs(float(corrected_value) - old_value)
            lines.append(
                f"| {label} | {statistic} | {old_value:.10f} | "
                f"{float(corrected_value):.10f} | {absolute_change:.10f} |"
            )

    threshold_unchanged = True
    for comparison in ("v1", "v2", "v2_minus_v1"):
        for metric in ("Precision", "Recall", "F1"):
            values = next(
                value for key, value in distributions.items()
                if key[0] == comparison and key[1] == metric
            )
            corrected = distribution_summary(values)
            old = old_rows[(comparison, metric)]
            for old_column, corrected_key in (
                ("bootstrap_mean", "mean"),
                ("ci_2_5_percentile", "lower"),
                ("ci_97_5_percentile", "upper"),
            ):
                if not math.isclose(
                    float(old[old_column]), float(corrected[corrected_key]),
                    rel_tol=0.0, abs_tol=5e-10,
                ):
                    threshold_unchanged = False
    lines.extend([
        "",
        f"Thresholded Precision/Recall/F1 bootstrap summaries unchanged: **{'PASS' if threshold_unchanged else 'FAIL'}**.",
        "",
    ])
    return "\n".join(lines), threshold_unchanged


def build_result_outputs(
    args: argparse.Namespace,
    distributions: dict,
    originals: dict,
    undefined_by_class: dict[int, int],
    any_undefined: int,
    validation_maximum: float,
    threshold_unchanged: bool,
    integrity_ok: bool,
) -> tuple[str, str]:
    columns = [
        "comparison", "metric", "original_point_estimate", "bootstrap_mean",
        "ci_2_5_percentile", "ci_97_5_percentile", "valid_replicates",
        "undefined_replicates", "replicates", "seed", "notes",
    ]
    rows = []
    for comparison, metric, notes in distributions:
        summary = distribution_summary(distributions[(comparison, metric, notes)])
        rows.append({
            "comparison": comparison,
            "metric": metric,
            "original_point_estimate": format_number(originals[(comparison, metric)]),
            "bootstrap_mean": format_number(summary["mean"]),
            "ci_2_5_percentile": format_number(summary["lower"]),
            "ci_97_5_percentile": format_number(summary["upper"]),
            "valid_replicates": summary["valid"],
            "undefined_replicates": summary["undefined"],
            "replicates": args.replicates,
            "seed": args.seed,
            "notes": notes,
        })

    # Construct CSV text in memory so all protected-input checks finish before writes.
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = "\ufeff" + buffer.getvalue()

    lines = [
        "# Corrected Paired Frame-Level Bootstrap 95% Confidence Intervals",
        "",
        "## Fixed method",
        "",
        "- Frozen frames per replicate: 60, sampled with replacement.",
        f"- Replicates: {args.replicates:,}",
        f"- Random seed: {args.seed}",
        "- Identical ordered frame samples were used for v1 and v2.",
        f"- Thresholded metrics: confidence {args.conf}, class-aware one-to-one matching, IoU {args.iou}.",
        f"- AP: literal occurrence expansion, confidence floor {args.ap_conf}, verified 101-point interpolation, IoU 0.50:0.95.",
        "- Fixed mAP classes: 0, 1, 4, 5, 6, 7.",
        "- Percentile interval: 2.5th and 97.5th percentiles using NumPy linear interpolation.",
        "- No model inference or training was run.",
        "",
        "## Expanded-AP validation",
        "",
        f"- Result: **{'PASS' if validation_maximum < 1e-12 else 'FAIL'}**",
        f"- Maximum absolute difference from literal verified evaluation: `{validation_maximum:.16g}`",
        "",
        "## Corrected bootstrap results",
        "",
        "| Comparison | Metric | Point estimate | Bootstrap mean | 2.5% | 97.5% | Valid | Undefined |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['original_point_estimate']} | "
            f"{row['bootstrap_mean']} | {row['ci_2_5_percentile']} | "
            f"{row['ci_97_5_percentile']} | {row['valid_replicates']} | "
            f"{row['undefined_replicates']} |"
        )
    lines.extend([
        "",
        "## Zero-GT fixed-class replicates",
        "",
        "| Class ID | Class name | Undefined replicates |",
        "|---:|---|---:|",
    ])
    for class_id in TARGET_CLASSES:
        lines.append(
            f"| {class_id} | {CLASS_NAMES[class_id]} | {undefined_by_class[class_id]} |"
        )
    lines.extend([
        "",
        f"Replicates with any fixed target class absent: **{any_undefined}**.",
        "Such replicates remain undefined; AP is not set to zero and the class set is not reduced.",
        "",
        "## Cross-checks",
        "",
        f"- Thresholded Precision/Recall/F1 summaries unchanged: **{'PASS' if threshold_unchanged else 'FAIL'}**",
        f"- Protected GT, images, predictions, weights, old scripts, and old results unchanged: **{'PASS' if integrity_ok else 'FAIL'}**",
        "- Google Docs were not accessed or updated.",
        "",
    ])
    return csv_text, "\n".join(lines)


def main() -> None:
    args = parse_arguments()
    if args.replicates != 10_000 or args.seed != 20_260_918:
        raise ValueError("Final protocol requires exactly 10,000 replicates and seed 20260918")
    if not math.isclose(args.conf, 0.35) or not math.isclose(args.iou, 0.50):
        raise ValueError("Final protocol requires conf=0.35 and IoU=0.50")
    if not math.isclose(args.ap_conf, 0.001):
        raise ValueError("Final protocol requires AP confidence floor 0.001")

    required = [
        prior.IMAGE_DIR, prior.LABEL_DIR, prior.AGGREGATE_RESULTS,
        prior.PER_CLASS_RESULTS, OLD_BOOTSTRAP_CSV, *prior.RAW_PREDICTION_DIRS.values(),
        *MODEL_PATHS,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for output in (OUTPUT_CSV, SUMMARY_MD, VALIDATION_MD, COMPARISON_MD):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    protected_files = [
        prior.AGGREGATE_RESULTS,
        prior.PER_CLASS_RESULTS,
        OLD_BOOTSTRAP_CSV,
        EVALUATION_DIR / "bootstrap_95ci_summary.md",
        EVALUATION_DIR / "per_frame_metrics.csv",
        EVALUATION_DIR / "bootstrap_code_audit.md",
        SCRIPTS_DIR / "bootstrap_frame_metrics.py",
        Path(__file__).resolve(),
        *MODEL_PATHS,
        *sorted(prior.IMAGE_DIR.iterdir()),
        *sorted(prior.LABEL_DIR.glob("*.txt")),
        *sorted(prior.RAW_PREDICTION_DIRS["v1"].glob("*.txt")),
        *sorted(prior.RAW_PREDICTION_DIRS["v2"].glob("*.txt")),
    ]
    protected_files = [path for path in protected_files if path.is_file()]
    before_hashes = prior.snapshot_hashes(protected_files)

    print("Loading frozen GT and saved raw predictions...", flush=True)
    _images, records_by_model = prior.load_frozen_records(args)
    locked = prior.read_locked_aggregate()
    _per_frame_rows, count_matrices = prior.calculate_per_frame(records_by_model, args)

    print("Validating literal expanded AP against verified evaluator...", flush=True)
    validation_text, validation_maximum = validate_expanded_ap(records_by_model)
    print(
        f"Expanded AP validation PASS; maximum absolute difference "
        f"{validation_maximum:.16g}", flush=True,
    )

    # Reconstruct and lock the unchanged original point estimates.
    point_ap = {}
    for model_name in ("v1", "v2"):
        totals = np.sum(count_matrices[model_name], axis=0)
        tp, fp, fn = (int(value) for value in totals)
        precision = prior.metric_value(tp, tp + fp)
        recall = prior.metric_value(tp, tp + fn)
        f1 = prior.f1_value(precision, recall)
        reconstructed = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "F1": f1,
        }
        for metric, value in reconstructed.items():
            if not np.isclose(value, locked[model_name][metric], atol=5e-8, rtol=0.0):
                raise RuntimeError(f"Point reconstruction failed: {model_name} {metric}")
        map50, map50_95, _class_ap = prior.reconstruct_point_ap(
            records_by_model[model_name]
        )
        point_ap[model_name] = (map50, map50_95)
        if not np.isclose(map50, locked[model_name]["mAP50"], atol=5e-7, rtol=0.0):
            raise RuntimeError(f"Point reconstruction failed: {model_name} mAP50")
        if not np.isclose(map50_95, locked[model_name]["mAP50_95"], atol=5e-7, rtol=0.0):
            raise RuntimeError(f"Point reconstruction failed: {model_name} mAP50_95")

    print("Generating one ordered paired bootstrap sample matrix...", flush=True)
    samples = generate_samples(args.replicates, args.seed)
    multiplicities = samples_to_multiplicities(samples)
    if not np.all(np.sum(multiplicities, axis=1) == 60):
        raise RuntimeError("A bootstrap replicate does not contain exactly 60 draws")

    distributions = {}
    originals = {}
    boot_threshold = {}
    for model_name in ("v1", "v2"):
        totals = multiplicities @ count_matrices[model_name]
        tp = totals[:, 0].astype(float)
        fp = totals[:, 1].astype(float)
        fn = totals[:, 2].astype(float)
        precision = prior.metric_value(tp, tp + fp)
        recall = prior.metric_value(tp, tp + fn)
        f1 = prior.f1_value(precision, recall)
        boot_threshold[model_name] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1,
        }
        locked_name = {"Precision": "precision", "Recall": "recall"}
        for metric in ("TP", "FP", "FN", "Precision", "Recall", "F1"):
            key = (model_name, metric, "Paired frame-level percentile bootstrap")
            distributions[key] = boot_threshold[model_name][metric]
            originals[(model_name, metric)] = locked[model_name][locked_name.get(metric, metric)]

    gt_reference_records = records_by_model["v1"]
    undefined_by_class = {}
    zero_gt_by_class = []
    for class_id in TARGET_CLASSES:
        gt_counts = np.asarray([
            sum(item["class_id"] == class_id for item in record["gt"])
            for record in gt_reference_records
        ], dtype=np.int16)
        zero_gt = (multiplicities @ gt_counts) == 0
        zero_gt_by_class.append(zero_gt)
        undefined_by_class[class_id] = int(np.sum(zero_gt))
    any_undefined = int(np.sum(np.any(np.stack(zero_gt_by_class, axis=1), axis=1)))

    bootstrap_class_ap = {"v1": {}, "v2": {}}
    for model_name in ("v1", "v2"):
        for class_id in TARGET_CLASSES:
            print(f"Preparing explicit AP events: {model_name} class {class_id}", flush=True)
            templates = prepare_occurrence_templates(records_by_model[model_name], class_id)
            bootstrap_class_ap[model_name][class_id] = expanded_bootstrap_ap(
                samples, templates, model_name, class_id, args.progress_every
            )
        stacked = np.stack([
            bootstrap_class_ap[model_name][class_id]
            for class_id in TARGET_CLASSES
        ], axis=1)
        valid = np.all(np.isfinite(stacked), axis=(1, 2))
        map50 = np.full(args.replicates, np.nan)
        map50_95 = np.full(args.replicates, np.nan)
        map50[valid] = np.mean(stacked[valid, :, 0], axis=1)
        map50_95[valid] = np.mean(stacked[valid], axis=(1, 2))
        distributions[(model_name, "mAP50", "Literal expanded occurrences; fixed classes 0,1,4,5,6,7")] = map50
        distributions[(model_name, "mAP50_95", "Literal expanded occurrences; fixed classes 0,1,4,5,6,7")] = map50_95
        originals[(model_name, "mAP50")] = locked[model_name]["mAP50"]
        originals[(model_name, "mAP50_95")] = locked[model_name]["mAP50_95"]

    for metric in ("Precision", "Recall", "F1"):
        difference = boot_threshold["v2"][metric] - boot_threshold["v1"][metric]
        distributions[("v2_minus_v1", metric, "Paired difference using identical ordered frame samples")] = difference
        locked_metric = {"Precision": "precision", "Recall": "recall", "F1": "F1"}[metric]
        originals[("v2_minus_v1", metric)] = (
            locked["v2"][locked_metric] - locked["v1"][locked_metric]
        )
    for metric in ("mAP50", "mAP50_95"):
        v1_values = next(
            value for key, value in distributions.items()
            if key[0] == "v1" and key[1] == metric
        )
        v2_values = next(
            value for key, value in distributions.items()
            if key[0] == "v2" and key[1] == metric
        )
        distributions[("v2_minus_v1", metric, "Paired literal-expanded AP difference")] = v2_values - v1_values
        originals[("v2_minus_v1", metric)] = locked["v2"][metric] - locked["v1"][metric]

    fp_signed = boot_threshold["v2"]["FP"] - boot_threshold["v1"]["FP"]
    fp_absolute_reduction = -fp_signed
    fp_percent_reduction = prior.metric_value(
        boot_threshold["v1"]["FP"] - boot_threshold["v2"]["FP"],
        boot_threshold["v1"]["FP"],
    ) * 100.0
    distributions[("v2_minus_v1", "FP", "Paired signed FP difference")] = fp_signed
    distributions[("v1_minus_v2", "FP_absolute_reduction", "Positive means fewer v2 false positives")] = fp_absolute_reduction
    distributions[("v1_to_v2", "FP_percent_reduction", "100 * (v1 FP - v2 FP) / v1 FP per replicate")] = fp_percent_reduction
    originals[("v2_minus_v1", "FP")] = locked["v2"]["FP"] - locked["v1"]["FP"]
    originals[("v1_minus_v2", "FP_absolute_reduction")] = locked["v1"]["FP"] - locked["v2"]["FP"]
    originals[("v1_to_v2", "FP_percent_reduction")] = (
        None if locked["v1"]["FP"] == 0
        else (locked["v1"]["FP"] - locked["v2"]["FP"]) / locked["v1"]["FP"] * 100.0
    )

    comparison_text, threshold_unchanged = build_comparison_markdown(
        distributions, originals
    )

    after_hashes = prior.snapshot_hashes(protected_files)
    integrity_ok = before_hashes == after_hashes
    if not integrity_ok:
        changed = [path for path in before_hashes if before_hashes[path] != after_hashes.get(path)]
        raise RuntimeError(f"Protected artifact changed: {changed}")

    csv_text, summary_text = build_result_outputs(
        args, distributions, originals, undefined_by_class, any_undefined,
        validation_maximum, threshold_unchanged, integrity_ok,
    )

    # Exclusive creation guarantees that prior or corrected results are never overwritten.
    for path, text in (
        (OUTPUT_CSV, csv_text),
        (SUMMARY_MD, summary_text),
        (VALIDATION_MD, validation_text),
        (COMPARISON_MD, comparison_text),
    ):
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)

    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {SUMMARY_MD}")
    print(f"Wrote: {VALIDATION_MD}")
    print(f"Wrote: {COMPARISON_MD}")
    print(f"Zero-GT fixed-class replicates: {any_undefined}")
    print(f"Thresholded Precision/Recall/F1 unchanged: {threshold_unchanged}")
    print(f"All protected-artifact integrity checks passed: {integrity_ok}")


if __name__ == "__main__":
    main()
