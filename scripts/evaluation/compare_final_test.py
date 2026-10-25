"""Evaluate frozen CJSJ human ground truth with v1 and v2 using identical settings.

This is a final-validation copy of compare_v1_v2_independent_test.py. It never
trains, changes model weights, or writes to the frozen image/label directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = ROOT / "data" / "private_final_test"
IMAGE_DIR = FROZEN_ROOT / "images"
LABEL_DIR = FROZEN_ROOT / "labels"
MANIFEST = ROOT / "data" / "private_final_test_freeze_manifest.csv"
PROTOCOL = ROOT / "docs" / "EVALUATION_PROTOCOL.md"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
MODELS = {
    "v1": ROOT / "weights" / "reef_ai_yolov12n_v1_controlled.pt",
    "v2": ROOT / "weights" / "reef_ai_yolov12n_v2_full_tank.pt",
}
CLASS_NAMES = [
    "Paracanthurus_hepatus",
    "Nemateleotris_magnifica",
    "Amphiprion_ocellaris_orange",
    "Amphiprion_ocellaris_black",
    "Dascyllus_aruanus",
    "Pomacentrus_moluccensis",
    "Chromis_viridis",
    "Monodactylus_argenteus",
]
COLORS = [
    (255, 140, 40), (255, 80, 210), (0, 150, 255), (150, 150, 150),
    (240, 240, 240), (0, 230, 255), (255, 210, 50), (70, 200, 110),
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final frozen 60-frame CJSJ v1-v2 evaluation")
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--labels", type=Path, default=LABEL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--v1-model", type=Path, default=MODELS["v1"])
    parser.add_argument("--v2-model", type=Path, default=MODELS["v2"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50,
                        help="Class-aware GT matching IoU threshold")
    parser.add_argument("--ap-conf", type=float, default=0.001,
                        help="Inference floor used only for confidence-ranked AP curves")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_manifest(manifest: Path, images_dir: Path, labels_dir: Path) -> dict[str, int]:
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 120:
        raise RuntimeError(f"Freeze manifest must contain 120 rows; found {len(rows)}")
    verified = Counter()
    for row in rows:
        file_type = row["file_type"]
        if file_type == "image":
            path = images_dir / row["filename"]
        elif file_type == "label":
            path = labels_dir / row["filename"]
        else:
            raise RuntimeError(f"Unknown manifest file type: {file_type}")
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file is missing: {path}")
        if path.stat().st_size != int(row["size_bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Frozen file failed manifest verification: {path}")
        verified[file_type] += 1
    if verified != Counter({"image": 60, "label": 60}):
        raise RuntimeError(f"Unexpected manifest composition: {dict(verified)}")
    return dict(verified)


def box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def read_ground_truth(label_file: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line_number, raw_line in enumerate(label_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw_line.strip()
        if not text:
            continue
        values = text.split()
        if len(values) != 5:
            raise ValueError(f"{label_file.name}:{line_number}: expected five values")
        class_id = int(values[0])
        center_x, center_y, box_width, box_height = map(float, values[1:])
        coords = (center_x, center_y, box_width, box_height)
        if class_id not in range(8) or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in coords):
            raise ValueError(f"{label_file.name}:{line_number}: invalid class or coordinate")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"{label_file.name}:{line_number}: non-positive box size")
        boxes.append({
            "class_id": class_id,
            "box": (
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ),
        })
    return boxes


def load_records(images_dir: Path, labels_dir: Path) -> list[dict]:
    images = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )
    if len(images) != 60:
        raise RuntimeError(f"Expected exactly 60 frozen images; found {len(images)}")
    records = []
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing frozen label: {label_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not decode frozen image: {image_path}")
        height, width = image.shape[:2]
        records.append({
            "path": image_path,
            "image": image,
            "width": width,
            "height": height,
            "gt": read_ground_truth(label_path, width, height),
        })
    return records


def extract_predictions(result) -> list[dict]:
    predictions = []
    if result.boxes is None:
        return predictions
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in range(8):
            continue
        predictions.append({
            "class_id": class_id,
            "score": float(box.conf[0].item()),
            "box": tuple(float(value) for value in box.xyxy[0].cpu().tolist()),
        })
    return predictions


def class_aware_match(predictions: list[dict], ground_truth: list[dict], threshold: float):
    matched_gt: set[int] = set()
    prediction_tp = [False] * len(predictions)
    for prediction_index in sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True):
        prediction = predictions[prediction_index]
        candidates = [
            (box_iou(prediction["box"], truth["box"]), gt_index)
            for gt_index, truth in enumerate(ground_truth)
            if gt_index not in matched_gt and truth["class_id"] == prediction["class_id"]
        ]
        if candidates:
            best_iou, best_gt = max(candidates)
            if best_iou >= threshold:
                prediction_tp[prediction_index] = True
                matched_gt.add(best_gt)
    return prediction_tp, matched_gt


def update_confusion_matrix(predictions: list[dict], ground_truth: list[dict], threshold: float,
                            matrix: np.ndarray) -> None:
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    for prediction_index in sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True):
        candidates = [
            (box_iou(predictions[prediction_index]["box"], truth["box"]), gt_index)
            for gt_index, truth in enumerate(ground_truth) if gt_index not in matched_gt
        ]
        if candidates:
            best_iou, best_gt = max(candidates)
            if best_iou >= threshold:
                matrix[ground_truth[best_gt]["class_id"], predictions[prediction_index]["class_id"]] += 1
                matched_gt.add(best_gt)
                matched_predictions.add(prediction_index)
    for gt_index, truth in enumerate(ground_truth):
        if gt_index not in matched_gt:
            matrix[truth["class_id"], 8] += 1
    for prediction_index, prediction in enumerate(predictions):
        if prediction_index not in matched_predictions:
            matrix[8, prediction["class_id"]] += 1


def average_precision(records: list[dict], class_id: int, threshold: float) -> float | None:
    gt_by_image: dict[int, list[tuple[float, ...]]] = {}
    total_gt = 0
    ranked_predictions = []
    for image_index, record in enumerate(records):
        gt_boxes = [item["box"] for item in record["gt"] if item["class_id"] == class_id]
        gt_by_image[image_index] = gt_boxes
        total_gt += len(gt_boxes)
        for prediction in record["predictions"]:
            if prediction["class_id"] == class_id:
                ranked_predictions.append((prediction["score"], image_index, prediction["box"]))
    if total_gt == 0:
        return None
    ranked_predictions.sort(key=lambda item: item[0], reverse=True)
    if not ranked_predictions:
        return 0.0
    matched: dict[int, set[int]] = defaultdict(set)
    tp, fp = [], []
    for _score, image_index, prediction_box in ranked_predictions:
        candidates = [
            (box_iou(prediction_box, gt_box), gt_index)
            for gt_index, gt_box in enumerate(gt_by_image[image_index])
            if gt_index not in matched[image_index]
        ]
        if candidates and max(candidates)[0] >= threshold:
            _iou, gt_index = max(candidates)
            matched[image_index].add(gt_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / total_gt
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(np.mean([
        np.max(precision[recall >= level], initial=0.0)
        for level in np.linspace(0.0, 1.0, 101)
    ]))


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return 0.0 if precision is not None and recall is not None else None
    return 2.0 * precision * recall / (precision + recall)


def write_raw_predictions(path: Path, predictions: list[dict], width: int, height: int) -> None:
    lines = []
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{prediction['class_id']} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f} {prediction['score']:.8f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def annotate(image: np.ndarray, predictions: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for prediction in predictions:
        class_id = prediction["class_id"]
        x1, y1, x2, y2 = map(round, prediction["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS[class_id], 2)
        cv2.putText(
            canvas,
            f"{class_id} {CLASS_NAMES[class_id]} {prediction['score']:.2f}",
            (max(0, x1), max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            COLORS[class_id],
            1,
            cv2.LINE_AA,
        )
    return canvas


def annotate_errors(image: np.ndarray, predictions: list[dict], ground_truth: list[dict],
                    prediction_tp: list[bool], matched_gt: set[int]) -> np.ndarray:
    canvas = image.copy()
    for index, prediction in enumerate(predictions):
        x1, y1, x2, y2 = map(round, prediction["box"])
        color = (0, 190, 0) if prediction_tp[index] else (0, 0, 255)
        text = "TP" if prediction_tp[index] else "FP"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)
    for gt_index, truth in enumerate(ground_truth):
        if gt_index not in matched_gt:
            x1, y1, x2, y2 = map(round, truth["box"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(canvas, "FN", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 165, 255), 1, cv2.LINE_AA)
    return canvas


def save_confusion_matrix(matrix: np.ndarray, model_name: str, model_dir: Path, threshold: float) -> None:
    labels = CLASS_NAMES + ["background"]
    with (model_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())
    figure, axis = plt.subplots(figsize=(10, 9))
    plotted = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(9), labels=labels, rotation=45, ha="right")
    axis.set_yticks(range(9), labels=labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Ground-truth class")
    axis.set_title(f"{model_name.upper()} Confusion Matrix (IoU >= {threshold:.2f})")
    for row in range(9):
        for column in range(9):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    figure.colorbar(plotted, ax=axis)
    figure.tight_layout()
    figure.savefig(model_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def evaluate_model(model_name: str, model_path: Path, base_records: list[dict], args: argparse.Namespace):
    model_dir = args.output / model_name
    raw_dir = model_dir / "raw_predictions"
    rendered_dir = model_dir / "rendered_predictions"
    fp_dir = model_dir / "false_positive_examples"
    fn_dir = model_dir / "false_negative_examples"
    logs_dir = model_dir / "logs"
    for directory in (raw_dir, rendered_dir, fp_dir, fn_dir, logs_dir):
        directory.mkdir(parents=True)
    log_lines = []

    def log(message: str) -> None:
        print(message, flush=True)
        log_lines.append(message)

    configuration = {
        "model_name": model_name,
        "model_path": str(model_path),
        "model_sha256_before": sha256(model_path),
        "images": str(args.images),
        "labels": str(args.labels),
        "image_count": 60,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "class_aware_matching_iou": args.iou,
        "ap_inference_floor": args.ap_conf,
        "ap_iou_thresholds": [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)],
        "device": str(args.device),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
    }
    (model_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )

    model = YOLO(str(model_path))
    records = []
    totals: Counter[str] = Counter()
    per_class = {class_id: Counter() for class_id in range(8)}
    matrix = np.zeros((9, 9), dtype=int)
    start = time.perf_counter()
    log(f"Evaluating {model_name}: {model_path}")
    for image_number, base in enumerate(base_records, 1):
        result = model.predict(
            source=base["image"],
            imgsz=args.imgsz,
            conf=args.ap_conf,
            device=args.device,
            verbose=False,
        )[0]
        all_predictions = extract_predictions(result)
        operating_predictions = [item for item in all_predictions if item["score"] >= args.conf]
        prediction_tp, matched_gt = class_aware_match(operating_predictions, base["gt"], args.iou)
        for prediction_index, prediction in enumerate(operating_predictions):
            key = "tp" if prediction_tp[prediction_index] else "fp"
            totals[key] += 1
            per_class[prediction["class_id"]][key] += 1
        for gt_index, truth in enumerate(base["gt"]):
            per_class[truth["class_id"]]["gt"] += 1
            if gt_index not in matched_gt:
                totals["fn"] += 1
                per_class[truth["class_id"]]["fn"] += 1
        update_confusion_matrix(operating_predictions, base["gt"], args.iou, matrix)
        write_raw_predictions(raw_dir / f"{base['path'].stem}.txt", all_predictions,
                              base["width"], base["height"])
        cv2.imwrite(str(rendered_dir / base["path"].name), annotate(base["image"], operating_predictions))
        error_image = annotate_errors(base["image"], operating_predictions, base["gt"],
                                      prediction_tp, matched_gt)
        if any(not value for value in prediction_tp):
            cv2.imwrite(str(fp_dir / base["path"].name), error_image)
        if len(matched_gt) < len(base["gt"]):
            cv2.imwrite(str(fn_dir / base["path"].name), error_image)
        records.append({"path": base["path"], "gt": base["gt"], "predictions": all_predictions})
        log(f"{image_number:02d}/60 {base['path'].name}: operating predictions={len(operating_predictions)}")

    thresholds = [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)]
    class_ap = {}
    for class_id in range(8):
        values = [average_precision(records, class_id, threshold) for threshold in thresholds]
        class_ap[class_id] = {
            "AP50": values[0],
            "AP50_95": None if values[0] is None else float(np.mean(values)),
        }
    present_classes = [class_id for class_id in range(8) if per_class[class_id]["gt"] > 0]
    precision = safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
    summary = {
        "model": model_name,
        "TP": totals["tp"],
        "FP": totals["fp"],
        "FN": totals["fn"],
        "precision": precision,
        "recall": recall,
        "F1": f1_score(precision, recall),
        "mAP50": float(np.mean([class_ap[class_id]["AP50"] for class_id in present_classes])),
        "mAP50_95": float(np.mean([class_ap[class_id]["AP50_95"] for class_id in present_classes])),
    }
    save_confusion_matrix(matrix, model_name, model_dir, args.iou)
    configuration["model_sha256_after"] = sha256(model_path)
    configuration["model_hash_unchanged"] = (
        configuration["model_sha256_before"] == configuration["model_sha256_after"]
    )
    configuration["elapsed_seconds"] = time.perf_counter() - start
    (model_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    log(f"Completed {model_name}: {summary}")
    (logs_dir / "evaluation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if not configuration["model_hash_unchanged"]:
        raise RuntimeError(f"Model weight changed during evaluation: {model_path}")
    return summary, per_class, class_ap


def csv_value(value) -> str | int:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def make_per_class_rows(results) -> list[dict]:
    rows = []
    for summary, counters, class_ap in results:
        for class_id in range(8):
            count = counters[class_id]
            present = count["gt"] > 0
            precision = safe_ratio(count["tp"], count["tp"] + count["fp"])
            recall = safe_ratio(count["tp"], count["tp"] + count["fn"]) if present else None
            f1 = f1_score(precision, recall) if present else None
            rows.append({
                "model": summary["model"],
                "class_id": class_id,
                "manuscript_name": CLASS_NAMES[class_id],
                "GT": count["gt"],
                "TP": count["tp"],
                "FP": count["fp"],
                "FN": count["fn"],
                "precision": csv_value(precision),
                "recall": csv_value(recall),
                "F1": csv_value(f1),
                "AP50": csv_value(class_ap[class_id]["AP50"]),
                "AP50_95": csv_value(class_ap[class_id]["AP50_95"]),
                "status": "present" if present else "absent_from_GT",
            })
    return rows


def write_final_outputs(results, args: argparse.Namespace, pre_verify: dict[str, int],
                        post_verify: dict[str, int]) -> None:
    aggregate_columns = ["model", "TP", "FP", "FN", "precision", "recall", "F1", "mAP50", "mAP50_95"]
    with (args.output / "aggregate_results.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_columns)
        writer.writeheader()
        for summary, _counters, _ap in results:
            writer.writerow({column: csv_value(summary[column]) for column in aggregate_columns})

    per_class_rows = make_per_class_rows(results)
    per_class_columns = [
        "model", "class_id", "manuscript_name", "GT", "TP", "FP", "FN",
        "precision", "recall", "F1", "AP50", "AP50_95", "status",
    ]
    with (args.output / "per_class_results.csv").open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_class_columns)
        writer.writeheader()
        writer.writerows(per_class_rows)

    v1 = results[0][0]
    v2 = results[1][0]
    fp_change = ((v2["FP"] - v1["FP"]) / v1["FP"] * 100.0) if v1["FP"] else None
    fp_reduction = ((v1["FP"] - v2["FP"]) / v1["FP"] * 100.0) if v1["FP"] else None
    lines = [
        "# CJSJ Final Validation 60: v1 vs v2",
        "",
        "## Final test design",
        "",
        "- 60 manually annotated, prediction-independent frames from the final full-tank recording.",
        "- Ground truth was audited and frozen before either model was evaluated.",
        f"- Identical settings: `imgsz={args.imgsz}`, confidence `{args.conf}`, class-aware matching IoU `{args.iou}`, device `{args.device}`.",
        f"- AP uses the verified prior evaluator's 101-point interpolation with inference floor `{args.ap_conf}`.",
        "- mAP averages include only positive-GT classes 0, 1, 4, 5, 6, and 7.",
        "- Classes 2 and 3 are excluded from AP averages; their thresholded predictions still count as false positives.",
        "",
        "## Frozen GT hash verification",
        "",
        f"- Before inference: {pre_verify['image']}/60 images and {pre_verify['label']}/60 labels matched the freeze manifest.",
        f"- After evaluation: {post_verify['image']}/60 images and {post_verify['label']}/60 labels matched the freeze manifest.",
        "",
        "## Aggregate results",
        "",
        "| Model | TP | FP | FN | Precision | Recall | F1 | mAP@50 | mAP@50:95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary, _counters, _ap in results:
        lines.append(
            f"| {summary['model']} | {summary['TP']} | {summary['FP']} | {summary['FN']} | "
            f"{summary['precision']:.6f} | {summary['recall']:.6f} | {summary['F1']:.6f} | "
            f"{summary['mAP50']:.6f} | {summary['mAP50_95']:.6f} |"
        )
    lines.extend([
        "",
        "## False-positive change",
        "",
        f"- Signed change from v1 to v2: {fp_change:.2f}%" if fp_change is not None else "- Signed change: N/A (v1 had zero false positives).",
        f"- False-positive reduction from v1 to v2: {fp_reduction:.2f}%" if fp_reduction is not None else "- Reduction: N/A.",
        "",
        "## Per-class results",
        "",
        "| Model | ID | Manuscript name | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50:95 | Status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in per_class_rows:
        lines.append(
            f"| {row['model']} | {row['class_id']} | {row['manuscript_name']} | {row['GT']} | "
            f"{row['TP']} | {row['FP']} | {row['FN']} | {row['precision']} | {row['recall']} | "
            f"{row['F1']} | {row['AP50']} | {row['AP50_95']} | {row['status']} |"
        )
    lines.extend([
        "",
        "## Absent-class false positives",
        "",
    ])
    for model_name in ("v1", "v2"):
        for class_id in (2, 3):
            row = next(item for item in per_class_rows if item["model"] == model_name and item["class_id"] == class_id)
            lines.append(f"- {model_name}, class {class_id} `{CLASS_NAMES[class_id]}`: {row['FP']} FP")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- The test contains 60 frames from one final-condition recording, so uncertainty and scene diversity remain limited.",
        "- Classes 2 and 3 have no positive GT instances; sensitivity and AP cannot be estimated for them.",
        "- AP is calculated by the project's verified custom evaluator and is conditional on predictions retained at the 0.001 inference floor.",
        "- Class 7 is matched by numeric ID. The manuscript name is `Monodactylus_argenteus`; historical model metadata may say `Platax_sp`.",
        "- Frames are sampled from a video and may not be statistically independent in the same way as recordings from different tanks or days.",
        "",
    ])
    (args.output / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")

    integrity = {
        "frozen_manifest_before": pre_verify,
        "frozen_manifest_after": post_verify,
        "label_hashes_unchanged": post_verify.get("label") == 60,
        "image_hashes_unchanged": post_verify.get("image") == 60,
        "training_run": False,
        "identical_settings_for_v1_v2": True,
        "model_hashes_unchanged": {
            name: json.loads((args.output / name / "configuration.json").read_text(encoding="utf-8"))["model_hash_unchanged"]
            for name in ("v1", "v2")
        },
    }
    (args.output / "integrity_report.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    if not 0.0 < args.ap_conf <= args.conf <= 1.0:
        raise ValueError("Require 0 < ap-conf <= conf <= 1")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("IoU must be between 0 and 1")
    models = {"v1": args.v1_model, "v2": args.v2_model}
    for required in (args.images, args.labels, args.manifest, PROTOCOL):
        if not required.exists():
            raise FileNotFoundError(required)
    for model_name, model_path in models.items():
        if not model_path.is_file():
            raise FileNotFoundError(f"{model_name} model is missing: {model_path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {args.output}")

    pre_verify = verify_frozen_manifest(args.manifest, args.images, args.labels)
    records = load_records(args.images, args.labels)
    args.output.mkdir(parents=True)
    root_configuration = {
        "script": str(Path(__file__).resolve()),
        "command": [sys.executable, *sys.argv],
        "settings": {
            "imgsz": args.imgsz,
            "confidence_threshold": args.conf,
            "class_aware_matching_iou": args.iou,
            "ap_inference_floor": args.ap_conf,
            "device": str(args.device),
        },
        "models": {name: str(path) for name, path in models.items()},
        "frozen_images": str(args.images),
        "frozen_labels": str(args.labels),
    }
    (args.output / "evaluation_configuration.json").write_text(
        json.dumps(root_configuration, indent=2), encoding="utf-8"
    )

    results = [evaluate_model(name, path, records, args) for name, path in models.items()]
    post_verify = verify_frozen_manifest(args.manifest, args.images, args.labels)
    write_final_outputs(results, args, pre_verify, post_verify)
    print("\nFinal CJSJ evaluation complete")
    print(f"Output: {args.output}")
    print(f"Frozen hashes after evaluation: {post_verify}")


if __name__ == "__main__":
    main()
