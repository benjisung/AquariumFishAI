"""Paired frame-level bootstrap for the frozen CJSJ 60-frame evaluation.

This script reads only frozen ground truth and already-saved raw predictions.
It does not load either YOLO model and does not perform inference or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compare_v1_v2_cjsj_final_validation_60 as verified  # noqa: E402


EVALUATION_DIR = ROOT / "outputs" / "evaluation"
FROZEN_DIR = ROOT / "data" / "private_final_test"
IMAGE_DIR = FROZEN_DIR / "images"
LABEL_DIR = FROZEN_DIR / "labels"
AGGREGATE_RESULTS = EVALUATION_DIR / "aggregate_results.csv"
PER_CLASS_RESULTS = EVALUATION_DIR / "per_class_results.csv"
RAW_PREDICTION_DIRS = {
    "v1": EVALUATION_DIR / "v1" / "raw_predictions",
    "v2": EVALUATION_DIR / "v2" / "raw_predictions",
}
OUTPUT_CSV = EVALUATION_DIR / "bootstrap_95ci.csv"
SUMMARY_MD = EVALUATION_DIR / "bootstrap_95ci_summary.md"
PER_FRAME_CSV = EVALUATION_DIR / "per_frame_metrics.csv"
TARGET_CLASSES = (0, 1, 4, 5, 6, 7)
IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)
CLASS_NAMES = verified.CLASS_NAMES


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired 60-frame bootstrap for frozen CJSJ results")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_918)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--ap-conf", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths}


def metric_value(numerator: np.ndarray | float, denominator: np.ndarray | float):
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan)
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return float(result) if result.ndim == 0 else result


def f1_value(precision, recall):
    precision_array = np.asarray(precision, dtype=float)
    recall_array = np.asarray(recall, dtype=float)
    denominator = precision_array + recall_array
    result = np.full(np.broadcast_shapes(precision_array.shape, recall_array.shape), np.nan)
    valid = np.isfinite(precision_array) & np.isfinite(recall_array)
    zero = valid & (denominator == 0)
    nonzero = valid & (denominator != 0)
    result[zero] = 0.0
    result[nonzero] = 2.0 * precision_array[nonzero] * recall_array[nonzero] / denominator[nonzero]
    return float(result) if result.ndim == 0 else result


def parse_raw_predictions(path: Path, width: int, height: int) -> list[dict]:
    predictions = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw_line.strip()
        if not text:
            continue
        values = text.split()
        if len(values) != 6:
            raise ValueError(f"{path.name}:{line_number}: expected six prediction values")
        class_id = int(values[0])
        center_x, center_y, box_width, box_height, score = map(float, values[1:])
        if class_id not in range(8) or not all(
            math.isfinite(value) for value in (center_x, center_y, box_width, box_height, score)
        ):
            raise ValueError(f"{path.name}:{line_number}: invalid prediction")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"{path.name}:{line_number}: confidence outside 0..1")
        predictions.append({
            "class_id": class_id,
            "score": score,
            "box": (
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ),
        })
    return predictions


def load_frozen_records(args: argparse.Namespace):
    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in verified.IMAGE_EXTENSIONS
    )
    if len(image_paths) != 60:
        raise RuntimeError(f"Expected 60 frozen images; found {len(image_paths)}")
    records_by_model = {"v1": [], "v2": []}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not decode frozen image: {image_path}")
        height, width = image.shape[:2]
        gt_path = LABEL_DIR / f"{image_path.stem}.txt"
        ground_truth = verified.read_ground_truth(gt_path, width, height)
        for model_name, prediction_dir in RAW_PREDICTION_DIRS.items():
            prediction_path = prediction_dir / f"{image_path.stem}.txt"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            predictions = parse_raw_predictions(prediction_path, width, height)
            if any(item["score"] + 1e-12 < args.ap_conf for item in predictions):
                raise RuntimeError(f"Prediction below declared AP floor in {prediction_path}")
            records_by_model[model_name].append({
                "frame_id": image_path.stem,
                "path": image_path,
                "gt": ground_truth,
                "predictions": predictions,
            })
    return image_paths, records_by_model


def calculate_per_frame(records_by_model: dict[str, list[dict]], args: argparse.Namespace):
    rows = []
    count_matrices = {}
    for model_name, records in records_by_model.items():
        counts = np.zeros((60, 3), dtype=np.int64)
        for frame_index, record in enumerate(records):
            operating = [item for item in record["predictions"] if item["score"] >= args.conf]
            prediction_tp, matched_gt = verified.class_aware_match(operating, record["gt"], args.iou)
            tp = sum(prediction_tp)
            fp = len(operating) - tp
            fn = len(record["gt"]) - len(matched_gt)
            counts[frame_index] = (tp, fp, fn)
            precision = metric_value(tp, tp + fp)
            recall = metric_value(tp, tp + fn)
            f1 = f1_value(precision, recall)
            rows.append({
                "frame_id": record["frame_id"],
                "model": model_name,
                "GT": len(record["gt"]),
                "predictions_conf_ge_0_35": len(operating),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": "N/A" if not np.isfinite(precision) else f"{precision:.10f}",
                "recall": "N/A" if not np.isfinite(recall) else f"{recall:.10f}",
                "F1": "N/A" if not np.isfinite(f1) else f"{f1:.10f}",
            })
        count_matrices[model_name] = counts
    return rows, count_matrices


def read_locked_aggregate() -> dict[str, dict[str, float]]:
    with AGGREGATE_RESULTS.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if {row["model"] for row in rows} != {"v1", "v2"}:
        raise RuntimeError("Locked aggregate results must contain v1 and v2")
    numeric_columns = ("TP", "FP", "FN", "precision", "recall", "F1", "mAP50", "mAP50_95")
    return {
        row["model"]: {column: float(row[column]) for column in numeric_columns}
        for row in rows
    }


def reconstruct_point_ap(records: list[dict]):
    class_ap = {}
    for class_id in TARGET_CLASSES:
        values = [
            verified.average_precision(records, class_id, float(threshold))
            for threshold in IOU_THRESHOLDS
        ]
        class_ap[class_id] = np.asarray(values, dtype=float)
    stacked = np.stack([class_ap[class_id] for class_id in TARGET_CLASSES], axis=0)
    return float(np.mean(stacked[:, 0])), float(np.mean(stacked)), class_ap


def prepare_ap_events(records: list[dict], class_id: int):
    gt_per_frame = np.zeros(60, dtype=np.int16)
    events = []
    for frame_index, record in enumerate(records):
        class_gt = [item for item in record["gt"] if item["class_id"] == class_id]
        gt_per_frame[frame_index] = len(class_gt)
        class_predictions = [
            item for item in record["predictions"] if item["class_id"] == class_id
        ]
        class_predictions = sorted(
            enumerate(class_predictions), key=lambda pair: pair[1]["score"], reverse=True
        )
        ordered_predictions = [item for _original_index, item in class_predictions]
        flags_by_threshold = []
        for threshold in IOU_THRESHOLDS:
            flags, _matched = verified.class_aware_match(
                ordered_predictions, class_gt, float(threshold)
            )
            flags_by_threshold.append(flags)
        for local_index, prediction in enumerate(ordered_predictions):
            events.append((
                -prediction["score"],
                frame_index,
                local_index,
                [flags_by_threshold[t][local_index] for t in range(len(IOU_THRESHOLDS))],
            ))
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    event_frames = np.asarray([item[1] for item in events], dtype=np.int16)
    tp_flags = np.asarray([item[3] for item in events], dtype=np.int8)
    if tp_flags.size == 0:
        tp_flags = np.zeros((0, len(IOU_THRESHOLDS)), dtype=np.int8)
    return gt_per_frame, event_frames, tp_flags


def bootstrap_ap(mult: np.ndarray, gt_per_frame: np.ndarray, event_frames: np.ndarray,
                 tp_flags: np.ndarray, batch_size: int) -> np.ndarray:
    """Return replicate x IoU-threshold AP using verified 101-point interpolation.

    Resampled duplicate frames are represented by integer event weights. This is
    algebraically equivalent to repeating each identical prediction/GT copy for
    the interpolated precision envelope and avoids dependence on bootstrap draw order.
    """
    replicate_count = mult.shape[0]
    threshold_count = tp_flags.shape[1]
    output = np.full((replicate_count, threshold_count), np.nan, dtype=np.float64)
    levels = np.linspace(0.0, 1.0, 101)
    if len(event_frames) == 0:
        total_gt = mult @ gt_per_frame.astype(np.int64)
        output[total_gt > 0, :] = 0.0
        return output

    for start in range(0, replicate_count, batch_size):
        stop = min(start + batch_size, replicate_count)
        batch_mult = mult[start:stop]
        total_gt = batch_mult @ gt_per_frame.astype(np.int64)
        weights = batch_mult[:, event_frames].astype(np.int32, copy=False)
        cumulative_predictions = np.cumsum(weights, axis=1, dtype=np.int32)
        valid_rows = np.flatnonzero(total_gt > 0)
        for threshold_index in range(threshold_count):
            weighted_tp = weights * tp_flags[:, threshold_index][None, :]
            cumulative_tp = np.cumsum(weighted_tp, axis=1, dtype=np.int32)
            precision = np.divide(
                cumulative_tp,
                cumulative_predictions,
                out=np.zeros_like(cumulative_tp, dtype=np.float64),
                where=cumulative_predictions != 0,
            )
            envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
            for row_index in valid_rows:
                targets = np.ceil(levels * total_gt[row_index] - 1e-12).astype(np.int32)
                indices = np.searchsorted(cumulative_tp[row_index], targets, side="left")
                values = np.zeros(101, dtype=np.float64)
                available = indices < len(event_frames)
                values[available] = envelope[row_index, indices[available]]
                output[start + row_index, threshold_index] = float(np.mean(values))
    return output


def generate_multiplicities(replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, 60, size=(replicates, 60), dtype=np.int16)
    multiplicities = np.zeros((replicates, 60), dtype=np.int16)
    for replicate_index in range(replicates):
        multiplicities[replicate_index] = np.bincount(
            sampled[replicate_index], minlength=60
        )
    return multiplicities


def summarize_distribution(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None, None, None, 0, len(values)
    lower, upper = np.percentile(finite, [2.5, 97.5], method="linear")
    return float(np.mean(finite)), float(lower), float(upper), len(finite), len(values) - len(finite)


def write_outputs(args: argparse.Namespace, per_frame_rows: list[dict], distributions: dict,
                  originals: dict, undefined_by_class: dict[int, int], any_undefined: int,
                  integrity_ok: bool, reconstruction_notes: list[str]) -> None:
    per_frame_columns = [
        "frame_id", "model", "GT", "predictions_conf_ge_0_35",
        "TP", "FP", "FN", "precision", "recall", "F1",
    ]
    with PER_FRAME_CSV.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_frame_columns)
        writer.writeheader()
        writer.writerows(per_frame_rows)

    ci_columns = [
        "comparison", "metric", "original_point_estimate", "bootstrap_mean",
        "ci_2_5_percentile", "ci_97_5_percentile", "valid_replicates",
        "undefined_replicates", "replicates", "seed", "notes",
    ]
    ci_rows = []
    for key, values in distributions.items():
        comparison, metric, notes = key
        mean, lower, upper, valid, undefined = summarize_distribution(values)
        ci_rows.append({
            "comparison": comparison,
            "metric": metric,
            "original_point_estimate": f"{originals[(comparison, metric)]:.10f}",
            "bootstrap_mean": "N/A" if mean is None else f"{mean:.10f}",
            "ci_2_5_percentile": "N/A" if lower is None else f"{lower:.10f}",
            "ci_97_5_percentile": "N/A" if upper is None else f"{upper:.10f}",
            "valid_replicates": valid,
            "undefined_replicates": undefined,
            "replicates": args.replicates,
            "seed": args.seed,
            "notes": notes,
        })
    with OUTPUT_CSV.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=ci_columns)
        writer.writeheader()
        writer.writerows(ci_rows)

    lines = [
        "# Paired Frame-Level Bootstrap 95% Confidence Intervals",
        "",
        "## Fixed method",
        "",
        f"- Frozen frames: 60",
        f"- Replicates: {args.replicates:,}",
        f"- Random seed: {args.seed}",
        "- Each replicate sampled 60 frame IDs with replacement.",
        "- The identical sampled frame multiplicities were used for v1 and v2.",
        f"- Thresholded metrics: confidence {args.conf}, class-aware one-to-one matching, IoU {args.iou}.",
        f"- AP: verified 101-point implementation, confidence floor {args.ap_conf}, IoU 0.50:0.95.",
        "- Manuscript mAP estimand is fixed to classes 0, 1, 4, 5, 6, and 7.",
        "- Percentile intervals use NumPy linear interpolation at the 2.5th and 97.5th percentiles.",
        "- No model inference or training was run.",
        "",
        "## Point-estimate reconstruction checks",
        "",
    ]
    lines.extend(f"- {note}" for note in reconstruction_notes)
    lines.extend([
        "",
        "## Bootstrap results",
        "",
        "| Comparison | Metric | Point estimate | Bootstrap mean | 2.5% | 97.5% | Valid | Undefined |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ci_rows:
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['original_point_estimate']} | "
            f"{row['bootstrap_mean']} | {row['ci_2_5_percentile']} | "
            f"{row['ci_97_5_percentile']} | {row['valid_replicates']} | "
            f"{row['undefined_replicates']} |"
        )
    lines.extend([
        "",
        "## Replicates with zero GT for a fixed target class",
        "",
        "| Class ID | Manuscript name | Undefined replicates |",
        "|---:|---|---:|",
    ])
    for class_id in TARGET_CLASSES:
        lines.append(f"| {class_id} | {CLASS_NAMES[class_id]} | {undefined_by_class[class_id]} |")
    lines.extend([
        "",
        f"Replicates with at least one undefined fixed-target class: **{any_undefined}**.",
        "When this occurs, the replicate-level mAP is marked undefined. The script does not average over a smaller class set and does not assign AP=0.",
        "",
        "## Integrity",
        "",
        f"- Existing aggregate/per-class results, raw predictions, frozen images, and frozen labels unchanged: **{'PASS' if integrity_ok else 'FAIL'}**",
        "- `aggregate_results.csv` and `per_class_results.csv` were read only.",
        "",
    ])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    if args.replicates != 10_000 or args.seed != 20_260_918:
        raise ValueError("Final protocol requires exactly 10,000 replicates and seed 20260918")
    if not math.isclose(args.conf, 0.35) or not math.isclose(args.iou, 0.50):
        raise ValueError("Final protocol requires conf=0.35 and IoU=0.50")
    if not math.isclose(args.ap_conf, 0.001):
        raise ValueError("Final protocol requires AP confidence floor 0.001")
    required = [IMAGE_DIR, LABEL_DIR, AGGREGATE_RESULTS, PER_CLASS_RESULTS, *RAW_PREDICTION_DIRS.values()]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for output in (OUTPUT_CSV, SUMMARY_MD, PER_FRAME_CSV):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    protected_files = [
        AGGREGATE_RESULTS,
        PER_CLASS_RESULTS,
        *sorted(IMAGE_DIR.iterdir()),
        *sorted(LABEL_DIR.glob("*.txt")),
        *sorted(RAW_PREDICTION_DIRS["v1"].glob("*.txt")),
        *sorted(RAW_PREDICTION_DIRS["v2"].glob("*.txt")),
    ]
    before_hashes = snapshot_hashes([path for path in protected_files if path.is_file()])

    print("Loading frozen GT and saved raw predictions...", flush=True)
    _image_paths, records_by_model = load_frozen_records(args)
    locked = read_locked_aggregate()
    per_frame_rows, count_matrices = calculate_per_frame(records_by_model, args)

    reconstruction_notes = []
    point_ap = {}
    for model_name in ("v1", "v2"):
        totals = np.sum(count_matrices[model_name], axis=0)
        tp, fp, fn = (int(value) for value in totals)
        precision = metric_value(tp, tp + fp)
        recall = metric_value(tp, tp + fn)
        f1 = f1_value(precision, recall)
        reconstructed = {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1}
        for metric in reconstructed:
            if not np.isclose(reconstructed[metric], locked[model_name][metric], atol=5e-8, rtol=0.0):
                raise RuntimeError(
                    f"Reconstructed {model_name} {metric}={reconstructed[metric]} does not match locked "
                    f"value {locked[model_name][metric]}"
                )
        print(f"Reconstructing original AP for {model_name}...", flush=True)
        map50, map50_95, _class_ap = reconstruct_point_ap(records_by_model[model_name])
        point_ap[model_name] = (map50, map50_95)
        if not np.isclose(map50, locked[model_name]["mAP50"], atol=5e-7, rtol=0.0):
            raise RuntimeError(f"Reconstructed {model_name} mAP50 differs from locked result")
        if not np.isclose(map50_95, locked[model_name]["mAP50_95"], atol=5e-7, rtol=0.0):
            raise RuntimeError(f"Reconstructed {model_name} mAP50_95 differs from locked result")
        reconstruction_notes.append(
            f"{model_name}: thresholded TP/FP/FN and mAP values reconstructed from saved predictions match the locked aggregate results."
        )

    print("Generating one paired bootstrap sampling matrix...", flush=True)
    multiplicities = generate_multiplicities(args.replicates, args.seed)
    distributions = {}
    originals = {}
    boot_threshold = {}
    for model_name in ("v1", "v2"):
        totals = multiplicities @ count_matrices[model_name]
        tp = totals[:, 0].astype(float)
        fp = totals[:, 1].astype(float)
        fn = totals[:, 2].astype(float)
        precision = metric_value(tp, tp + fp)
        recall = metric_value(tp, tp + fn)
        f1 = f1_value(precision, recall)
        boot_threshold[model_name] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1,
        }
        locked_name = {"Precision": "precision", "Recall": "recall"}
        for metric in ("TP", "FP", "FN", "Precision", "Recall", "F1"):
            key = (model_name, metric, "Paired frame-level percentile bootstrap")
            distributions[key] = boot_threshold[model_name][metric]
            originals[(model_name, metric)] = locked[model_name][locked_name.get(metric, metric)]

    undefined_by_class = {}
    gt_reference_records = records_by_model["v1"]
    for class_id in TARGET_CLASSES:
        gt_counts = np.asarray([
            sum(item["class_id"] == class_id for item in record["gt"])
            for record in gt_reference_records
        ], dtype=np.int16)
        undefined_by_class[class_id] = int(np.sum((multiplicities @ gt_counts) == 0))
    any_undefined = int(np.sum(np.any(np.stack([
        (multiplicities @ np.asarray([
            sum(item["class_id"] == class_id for item in record["gt"])
            for record in gt_reference_records
        ], dtype=np.int16)) == 0
        for class_id in TARGET_CLASSES
    ], axis=1), axis=1)))

    bootstrap_class_ap = {"v1": {}, "v2": {}}
    for model_name in ("v1", "v2"):
        for class_id in TARGET_CLASSES:
            print(f"Bootstrap AP: {model_name} class {class_id}...", flush=True)
            gt_per_frame, event_frames, tp_flags = prepare_ap_events(
                records_by_model[model_name], class_id
            )
            bootstrap_class_ap[model_name][class_id] = bootstrap_ap(
                multiplicities, gt_per_frame, event_frames, tp_flags, args.batch_size
            )
        stacked = np.stack([
            bootstrap_class_ap[model_name][class_id]
            for class_id in TARGET_CLASSES
        ], axis=1)
        valid = np.all(np.isfinite(stacked), axis=(1, 2))
        map50 = np.full(args.replicates, np.nan)
        map50_95 = np.full(args.replicates, np.nan)
        map50[valid] = np.mean(stacked[valid, :, 0], axis=1)
        map50_95[valid] = np.mean(stacked[valid, :, :], axis=(1, 2))
        distributions[(model_name, "mAP50", "Fixed-class estimand: classes 0,1,4,5,6,7")] = map50
        distributions[(model_name, "mAP50_95", "Fixed-class estimand: classes 0,1,4,5,6,7")] = map50_95
        originals[(model_name, "mAP50")] = locked[model_name]["mAP50"]
        originals[(model_name, "mAP50_95")] = locked[model_name]["mAP50_95"]

    for metric in ("Precision", "Recall", "F1"):
        difference = boot_threshold["v2"][metric] - boot_threshold["v1"][metric]
        distributions[("v2_minus_v1", metric, "Paired difference using identical frame multiplicities")] = difference
        locked_metric = {"Precision": "precision", "Recall": "recall", "F1": "F1"}[metric]
        originals[("v2_minus_v1", metric)] = locked["v2"][locked_metric] - locked["v1"][locked_metric]
    for metric in ("mAP50", "mAP50_95"):
        difference = distributions[("v2", metric, "Fixed-class estimand: classes 0,1,4,5,6,7")] - distributions[("v1", metric, "Fixed-class estimand: classes 0,1,4,5,6,7")]
        distributions[("v2_minus_v1", metric, "Paired difference; undefined when fixed-class mAP is undefined")] = difference
        originals[("v2_minus_v1", metric)] = locked["v2"][metric] - locked["v1"][metric]

    fp_signed = boot_threshold["v2"]["FP"] - boot_threshold["v1"]["FP"]
    fp_absolute_reduction = -fp_signed
    fp_percent_reduction = metric_value(
        boot_threshold["v1"]["FP"] - boot_threshold["v2"]["FP"],
        boot_threshold["v1"]["FP"],
    ) * 100.0
    distributions[("v2_minus_v1", "FP", "Paired signed FP difference")] = fp_signed
    distributions[("v1_minus_v2", "FP_absolute_reduction", "Positive means fewer v2 false positives")] = fp_absolute_reduction
    distributions[("v1_to_v2", "FP_percent_reduction", "100 * (v1 FP - v2 FP) / v1 FP per replicate")] = fp_percent_reduction
    originals[("v2_minus_v1", "FP")] = locked["v2"]["FP"] - locked["v1"]["FP"]
    originals[("v1_minus_v2", "FP_absolute_reduction")] = locked["v1"]["FP"] - locked["v2"]["FP"]
    originals[("v1_to_v2", "FP_percent_reduction")] = (
        (locked["v1"]["FP"] - locked["v2"]["FP"]) / locked["v1"]["FP"] * 100.0
    )

    after_hashes = snapshot_hashes([path for path in protected_files if path.is_file()])
    integrity_ok = before_hashes == after_hashes
    if not integrity_ok:
        raise RuntimeError("A protected GT, prediction, or existing result file changed")

    write_outputs(
        args, per_frame_rows, distributions, originals, undefined_by_class,
        any_undefined, integrity_ok, reconstruction_notes,
    )
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {SUMMARY_MD}")
    print(f"Wrote: {PER_FRAME_CSV}")
    print(f"Replicates with any fixed target class absent: {any_undefined}")


if __name__ == "__main__":
    main()
