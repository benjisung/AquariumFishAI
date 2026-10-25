from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import math
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Defaults and class names
# ---------------------------------------------------------------------------
DEFAULT_SOURCE = None
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "weights" / "reef_ai_yolov12n_v2_full_tank.pt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tracking_analysis"
CHINESE_FONT_FILE = Path(r"C:\Windows\Fonts\msjh.ttc")

COMMON_NAMES = {
    0: "Blue tang",
    1: "Firefish",
    2: "Orange clownfish",
    3: "Black clownfish",
    4: "Humbug damselfish",
    5: "Lemon damselfish",
    6: "Blue-green chromis",
    7: "Silver moony",
}

SCIENTIFIC_NAMES = {
    0: "Paracanthurus_hepatus",
    1: "Nemateleotris_magnifica",
    2: "Amphiprion_ocellaris_orange",
    3: "Amphiprion_ocellaris_black",
    4: "Dascyllus_aruanus",
    5: "Pomacentrus_moluccensis",
    6: "Chromis_viridis",
    7: "Monodactylus_argenteus",
}

CHINESE_NAMES = {
    0: "藍倒吊",
    1: "雷達魚",
    2: "橘小丑",
    3: "黑小丑",
    4: "黑白雀",
    5: "黃雀鯛",
    6: "青魔",
    7: "蝙蝠魚",
}

BOX_COLORS = {
    0: (255, 170, 40),
    1: (255, 100, 210),
    2: (0, 150, 255),
    3: (170, 170, 170),
    4: (240, 240, 240),
    5: (0, 230, 255),
    6: (255, 220, 60),
    7: (80, 200, 120),
}

TRACKING_COLUMNS = [
    "frame_index",
    "time_seconds",
    "track_id",
    "class_id",
    "display_species_name",
    "scientific_species_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "box_width",
    "box_height",
]


def parse_hidden_classes(value):
    """Convert a comma-separated class list into a set of IDs."""
    if value is None or not value.strip():
        return set()
    try:
        class_ids = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--hide_classes must contain comma-separated integer class IDs"
        ) from error
    invalid_ids = sorted(class_id for class_id in class_ids if class_id not in COMMON_NAMES)
    if invalid_ids:
        raise argparse.ArgumentTypeError(
            f"Unknown class IDs in --hide_classes: {invalid_ids}"
        )
    return class_ids


def parse_analysis_roi(value):
    """Convert normalized x1,y1,x2,y2 text into a validated ROI tuple."""
    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--analysis_roi must contain four numbers: x1,y1,x2,y2"
        ) from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError(
            "--analysis_roi must contain exactly four values"
        )
    x1, y1, x2, y2 = coordinates
    if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in coordinates):
        raise argparse.ArgumentTypeError("Analysis ROI values must be between 0 and 1")
    if x1 >= x2 or y1 >= y2:
        raise argparse.ArgumentTypeError("Analysis ROI requires x1 < x2 and y1 < y2")
    return coordinates


def parse_arguments():
    """Read analysis settings from the command line."""
    parser = argparse.ArgumentParser(
        description="Run YOLO tracking and quantitative fish movement analysis."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--label_lang", choices=("en", "zh"), default="en")
    parser.add_argument("--show_conf", action="store_true")
    parser.add_argument("--show_id", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--box_thickness", type=int, default=2)
    parser.add_argument(
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write annotated_tracking_video.mp4 (default: enabled)",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=None,
        help="Annotated-video FPS; default uses source FPS",
    )
    parser.add_argument(
        "--hide_classes",
        type=parse_hidden_classes,
        default=set(),
        help="Comma-separated class IDs excluded from every output",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional maximum number of source frames to analyze",
    )
    parser.add_argument("--min_track_frames", type=int, default=20)
    parser.add_argument("--min_track_duration", type=float, default=0.5)
    parser.add_argument(
        "--filter_summary_only",
        action="store_true",
        help="Keep unfiltered analysis-ROI detections in per-second count CSVs",
    )
    parser.add_argument(
        "--trajectory_labels",
        choices=("none", "id", "species"),
        default="species",
    )
    parser.add_argument("--trajectory_min_points", type=int, default=5)
    parser.add_argument("--trajectory_line_thickness", type=float, default=2.0)
    parser.add_argument("--max_trajectory_labels", type=int, default=12)
    parser.add_argument("--smooth_trajectories", action="store_true")
    parser.add_argument("--heatmap_alpha", type=float, default=0.55)
    parser.add_argument(
        "--analysis_roi",
        type=parse_analysis_roi,
        default=None,
        metavar="X1,Y1,X2,Y2",
    )
    parser.add_argument(
        "--clean_video",
        action="store_true",
        help="Use English species-only presentation labels without IDs",
    )
    parser.add_argument(
        "--merge_tracklets",
        action="store_true",
        help="Conservatively merge fragmented valid tracklets",
    )
    parser.add_argument("--merge_max_gap_seconds", type=float, default=0.75)
    parser.add_argument("--merge_max_distance_pixels", type=float, default=180.0)
    parser.add_argument(
        "--merge_same_species_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require majority-vote species agreement when merging (default: enabled)",
    )
    parser.add_argument("--merge_min_iou", type=float, default=0.0)
    return parser.parse_args()


def validate_arguments(args):
    """Validate paths and numeric settings before opening the video."""
    if not args.source.is_file():
        raise FileNotFoundError(f"Input video was not found: {args.source}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model was not found: {args.model}")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not args.tracker.strip():
        raise ValueError("--tracker cannot be empty")
    if args.box_thickness <= 0:
        raise ValueError("--box_thickness must be greater than 0")
    if args.output_fps is not None and args.output_fps <= 0:
        raise ValueError("--output_fps must be greater than 0")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max_frames must be greater than 0")
    if args.min_track_frames <= 0:
        raise ValueError("--min_track_frames must be greater than 0")
    if args.min_track_duration < 0:
        raise ValueError("--min_track_duration cannot be negative")
    if args.trajectory_min_points <= 0:
        raise ValueError("--trajectory_min_points must be greater than 0")
    if args.trajectory_line_thickness <= 0:
        raise ValueError("--trajectory_line_thickness must be greater than 0")
    if args.max_trajectory_labels < 0:
        raise ValueError("--max_trajectory_labels cannot be negative")
    if not 0.0 <= args.heatmap_alpha <= 1.0:
        raise ValueError("--heatmap_alpha must be between 0 and 1")
    if args.merge_max_gap_seconds < 0:
        raise ValueError("--merge_max_gap_seconds cannot be negative")
    if args.merge_max_distance_pixels < 0:
        raise ValueError("--merge_max_distance_pixels cannot be negative")
    if not 0.0 <= args.merge_min_iou <= 1.0:
        raise ValueError("--merge_min_iou must be between 0 and 1")


def open_source_video(source):
    """Open the source video and return its metadata."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid FPS or frame dimensions")
    return capture, source_fps, frame_width, frame_height, total_frames


def create_video_writer(output_file, fps, frame_width, frame_height):
    """Create the optional MP4 annotated-video writer."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video: {output_file}")
    return writer


def load_chinese_font(frame_width):
    """Load Microsoft JhengHei for Chinese video labels."""
    if not CHINESE_FONT_FILE.is_file():
        raise FileNotFoundError(
            f"Chinese label font was not found: {CHINESE_FONT_FILE}"
        )
    font_size = max(12, round(18 * frame_width / 1280))
    return ImageFont.truetype(str(CHINESE_FONT_FILE), font_size)


def zone_for_x(center_x, frame_width):
    """Assign a center point to the left, center, or right vertical zone."""
    if center_x < frame_width / 3:
        return "left"
    if center_x < frame_width * 2 / 3:
        return "center"
    return "right"


def center_is_inside_analysis_roi(detection, analysis_roi, frame_width, frame_height):
    """Return True when a detection center is inside the normalized analysis ROI."""
    if analysis_roi is None:
        return True
    x1, y1, x2, y2 = analysis_roi
    normalized_x = detection["center_x"] / frame_width
    normalized_y = detection["center_y"] / frame_height
    return x1 <= normalized_x <= x2 and y1 <= normalized_y <= y2


def make_label_text(detection, display_name, args):
    """Build the compact video label requested by the display options."""
    parts = []
    if args.show_id and not args.clean_video and detection["track_id"] is not None:
        parts.append(f"ID{detection['track_id']}")
    parts.append(display_name)
    if args.show_conf:
        parts.append(f"{detection['confidence']:.2f}")
    return " ".join(parts)


def draw_english_overlay(frame, detections, args):
    """Draw fast English boxes, labels, and optional current-frame counts."""
    for detection in detections:
        class_id = detection["class_id"]
        color = BOX_COLORS.get(class_id, (0, 255, 0))
        x1, y1, x2, y2 = detection["box_int"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.box_thickness)

        label = make_label_text(detection, COMMON_NAMES[class_id], args)
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )
        label_height = text_size[1] + baseline + 4
        label_y = y1 - label_height if y1 >= label_height else y1
        cv2.rectangle(
            frame,
            (x1, label_y),
            (x1 + text_size[0] + 6, label_y + label_height),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y + text_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if args.stats:
        draw_english_stats_panel(frame, detections)
    return frame


def draw_english_stats_panel(frame, detections):
    """Draw a compact dark English current-frame species panel."""
    counts = Counter(detection["class_id"] for detection in detections)
    if not counts:
        return

    lines = ["Current detections"] + [
        f"{COMMON_NAMES[class_id]} x{counts[class_id]}" for class_id in sorted(counts)
    ]
    font_scale = 0.48
    line_height = 22
    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for line in lines
    ]
    panel_width = max(widths) + 18
    panel_height = len(lines) * line_height + 12
    frame_width = frame.shape[1]
    x1 = max(0, frame_width - panel_width - 10)
    y1 = 10

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, y1),
        (x1 + panel_width, y1 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        color = (80, 230, 255) if index == 0 else (245, 245, 245)
        cv2.putText(
            frame,
            line,
            (x1 + 9, y1 + 20 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_chinese_overlay(frame, detections, font, args):
    """Draw boxes with OpenCV and all Chinese text in one Pillow conversion."""
    label_items = []
    frame_height, frame_width = frame.shape[:2]
    for detection in detections:
        class_id = detection["class_id"]
        color = BOX_COLORS.get(class_id, (0, 255, 0))
        x1, y1, x2, y2 = detection["box_int"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, args.box_thickness)
        label = make_label_text(detection, CHINESE_NAMES[class_id], args)
        label_items.append((x1, y1, label, color))

    base_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert(
        "RGBA"
    )
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x1, y1, label, bgr_color in label_items:
        bounds = draw.textbbox((0, 0), label, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        background_width = text_width + 6
        background_height = text_height + 4
        label_x = min(x1, max(0, frame_width - background_width))
        label_y = y1 - background_height - 2
        if label_y < 0:
            label_y = min(frame_height - background_height, y1 + 2)
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0], 255)
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + background_width,
                label_y + background_height,
            ),
            fill=(0, 0, 0, 210),
            outline=rgb_color,
            width=1,
        )
        draw.text(
            (label_x + 3, label_y + 2 - bounds[1]),
            label,
            font=font,
            fill=rgb_color,
        )

    if args.stats and detections:
        counts = Counter(detection["class_id"] for detection in detections)
        lines = ["目前偵測"] + [
            f"{CHINESE_NAMES[class_id]} x{counts[class_id]}"
            for class_id in sorted(counts)
        ]
        sizes = []
        for line in lines:
            bounds = draw.textbbox((0, 0), line, font=font)
            sizes.append((bounds[2] - bounds[0], bounds[3] - bounds[1], bounds[1]))
        padding = 8
        gap = 3
        panel_width = max(width for width, _height, _top in sizes) + padding * 2
        panel_height = (
            sum(height for _width, height, _top in sizes)
            + gap * (len(lines) - 1)
            + padding * 2
        )
        panel_x = max(0, frame_width - panel_width - 10)
        panel_y = 10
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            radius=5,
            fill=(0, 0, 0, 175),
            outline=(215, 215, 215, 190),
            width=1,
        )
        text_y = panel_y + padding
        for index, (line, (_width, height, top)) in enumerate(zip(lines, sizes)):
            color = (255, 230, 120, 255) if index == 0 else (255, 255, 255, 255)
            draw.text(
                (panel_x + padding, text_y - top),
                line,
                font=font,
                fill=color,
            )
            text_y += height + gap

    combined = Image.alpha_composite(base_image, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(combined), cv2.COLOR_RGB2BGR)


def extract_detections(result, frame_width, frame_height, hidden_classes):
    """Convert one Ultralytics result into beginner-readable dictionaries."""
    detections = []
    if result.boxes is None:
        return detections

    for detection_index, box in enumerate(result.boxes):
        class_id = int(box.cls[0].item())
        if class_id in hidden_classes or class_id not in COMMON_NAMES:
            continue

        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].cpu().tolist())
        x1 = max(0.0, min(frame_width - 1.0, x1))
        y1 = max(0.0, min(frame_height - 1.0, y1))
        x2 = max(0.0, min(frame_width - 1.0, x2))
        y2 = max(0.0, min(frame_height - 1.0, y2))
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)

        track_id = None
        if box.id is not None:
            track_id = int(box.id[0].item())

        detections.append(
            {
                "detection_index": detection_index,
                "track_id": track_id,
                "class_id": class_id,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "box_width": box_width,
                "box_height": box_height,
                "box_area": box_width * box_height,
                "zone": zone_for_x(center_x, frame_width),
                "box_int": tuple(round(value) for value in (x1, y1, x2, y2)),
            }
        )
    return detections


def write_tracking_row(csv_writer, detection, frame_index, time_seconds, display_names):
    """Write one detection row immediately to tracking_data.csv."""
    class_id = detection["class_id"]
    csv_writer.writerow(
        {
            "frame_index": frame_index,
            "time_seconds": f"{time_seconds:.6f}",
            "track_id": "" if detection["track_id"] is None else detection["track_id"],
            "class_id": class_id,
            "display_species_name": display_names[class_id],
            "scientific_species_name": SCIENTIFIC_NAMES[class_id],
            "confidence": f"{detection['confidence']:.6f}",
            "x1": f"{detection['x1']:.3f}",
            "y1": f"{detection['y1']:.3f}",
            "x2": f"{detection['x2']:.3f}",
            "y2": f"{detection['y2']:.3f}",
            "center_x": f"{detection['center_x']:.3f}",
            "center_y": f"{detection['center_y']:.3f}",
            "box_width": f"{detection['box_width']:.3f}",
            "box_height": f"{detection['box_height']:.3f}",
        }
    )


def update_per_second_counts(per_second_species, per_second_zones, second, detection, frame_index):
    """Record unique tracked fish seen in each species and zone per second."""
    track_id = detection["track_id"]
    if track_id is None:
        # Approximate anonymous fish slots within this second. Using the
        # detection position instead of every frame avoids inflating a single
        # untracked fish into dozens of fish at normal video frame rates.
        identity = ("untracked_slot", detection["detection_index"])
    else:
        identity = ("track", track_id)
    per_second_species[second][detection["class_id"]].add(identity)
    per_second_zones[second][detection["zone"]].add(identity)


def majority_species_id(points):
    """Return the majority-vote class, breaking ties by mean confidence."""
    class_counts = Counter(point["class_id"] for point in points)
    highest_count = max(class_counts.values())
    candidates = [
        class_id for class_id, count in class_counts.items() if count == highest_count
    ]
    if len(candidates) == 1:
        return candidates[0]

    mean_confidences = {}
    for class_id in candidates:
        confidences = [
            point["confidence"] for point in points if point["class_id"] == class_id
        ]
        mean_confidences[class_id] = sum(confidences) / len(confidences)
    return max(candidates, key=lambda class_id: mean_confidences[class_id])


def build_track_summaries(track_points, source_fps):
    """Calculate species-smoothed movement and zone results per track."""
    summaries = []
    zone_rows = []
    final_species_by_track = {}
    frame_duration = 1.0 / source_fps

    for track_id in sorted(track_points):
        points = sorted(track_points[track_id], key=lambda point: point["time_seconds"])
        final_species_id = majority_species_id(points)
        final_species_by_track[track_id] = final_species_id
        first_time = points[0]["time_seconds"]
        last_time = points[-1]["time_seconds"]
        duration = max(0.0, last_time - first_time)

        path_length = 0.0
        speeds = []
        for previous, current in zip(points, points[1:]):
            distance = math.hypot(
                current["center_x"] - previous["center_x"],
                current["center_y"] - previous["center_y"],
            )
            delta_time = current["time_seconds"] - previous["time_seconds"]
            path_length += distance
            if delta_time > 0:
                speeds.append(distance / delta_time)

        zone_counts = Counter(point["zone"] for point in points)
        most_used_zone = max(
            ("left", "center", "right"),
            key=lambda zone: zone_counts[zone],
        )
        zone_times = {
            zone: zone_counts[zone] * frame_duration
            for zone in ("left", "center", "right")
        }

        summaries.append(
            {
                "track_id": track_id,
                "final_species_id": final_species_id,
                "final_species_common_name": COMMON_NAMES[final_species_id],
                "final_species_scientific_name": SCIENTIFIC_NAMES[final_species_id],
                "first_seen_time": first_time,
                "last_seen_time": last_time,
                "duration_seconds": duration,
                "num_frames_seen": len(points),
                "mean_confidence": sum(point["confidence"] for point in points)
                / len(points),
                "total_path_length_pixels": path_length,
                "average_speed_pixels_per_second": (
                    path_length / duration if duration > 0 else 0.0
                ),
                "max_speed_pixels_per_second": max(speeds, default=0.0),
                "mean_box_area": sum(point["box_area"] for point in points)
                / len(points),
                "zone_most_used": most_used_zone,
            }
        )
        zone_rows.append(
            {
                "track_id": track_id,
                "species": COMMON_NAMES[final_species_id],
                "left_time_seconds": zone_times["left"],
                "center_time_seconds": zone_times["center"],
                "right_time_seconds": zone_times["right"],
                "most_used_zone": most_used_zone,
            }
        )

    return summaries, zone_rows, final_species_by_track


def filter_valid_tracks(track_points, min_frames, min_duration):
    """Keep tracks that satisfy both frame-count and elapsed-duration thresholds."""
    valid_tracks = {}
    for track_id, points in track_points.items():
        if not points:
            continue
        first_time = min(point["time_seconds"] for point in points)
        last_time = max(point["time_seconds"] for point in points)
        duration = max(0.0, last_time - first_time)
        if len(points) >= min_frames and duration >= min_duration:
            valid_tracks[track_id] = points
    return valid_tracks


def box_iou(first_point, second_point):
    """Calculate endpoint box intersection-over-union for optional merge gating."""
    intersection_x1 = max(first_point["x1"], second_point["x1"])
    intersection_y1 = max(first_point["y1"], second_point["y1"])
    intersection_x2 = min(first_point["x2"], second_point["x2"])
    intersection_y2 = min(first_point["y2"], second_point["y2"])
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    union_area = first_point["box_area"] + second_point["box_area"] - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def merge_valid_tracklets(valid_track_points, final_species_by_track, args):
    """Greedily chain each ended tracklet to its nearest valid next tracklet."""
    if not args.merge_tracklets:
        identity_mapping = {track_id: track_id for track_id in valid_track_points}
        return dict(valid_track_points), identity_mapping, 0

    ordered_track_ids = sorted(
        valid_track_points,
        key=lambda track_id: (
            valid_track_points[track_id][0]["time_seconds"],
            track_id,
        ),
    )
    unassigned = set(ordered_track_ids)
    merged_tracks = {}
    original_to_merged = {}
    merged_track_id = 1

    while unassigned:
        seed_id = min(
            unassigned,
            key=lambda track_id: (
                valid_track_points[track_id][0]["time_seconds"],
                track_id,
            ),
        )
        chain_ids = [seed_id]
        unassigned.remove(seed_id)
        tail_id = seed_id

        while True:
            tail_points = valid_track_points[tail_id]
            tail_last = tail_points[-1]
            candidates = []

            for candidate_id in unassigned:
                candidate_points = valid_track_points[candidate_id]
                candidate_first = candidate_points[0]
                time_gap = (
                    candidate_first["time_seconds"] - tail_last["time_seconds"]
                )
                if time_gap <= 0 or time_gap > args.merge_max_gap_seconds:
                    continue

                distance = math.hypot(
                    candidate_first["center_x"] - tail_last["center_x"],
                    candidate_first["center_y"] - tail_last["center_y"],
                )
                if distance > args.merge_max_distance_pixels:
                    continue

                if (
                    args.merge_same_species_only
                    and final_species_by_track[tail_id]
                    != final_species_by_track[candidate_id]
                ):
                    continue

                endpoint_iou = box_iou(tail_last, candidate_first)
                if endpoint_iou < args.merge_min_iou:
                    continue

                candidates.append((distance, time_gap, candidate_id))

            if not candidates:
                break

            _distance, _time_gap, next_id = min(candidates)
            chain_ids.append(next_id)
            unassigned.remove(next_id)
            tail_id = next_id

        merged_points = []
        for original_id in chain_ids:
            original_to_merged[original_id] = merged_track_id
            merged_points.extend(valid_track_points[original_id])
        merged_tracks[merged_track_id] = sorted(
            merged_points,
            key=lambda point: (point["time_seconds"], point["frame_index"]),
        )
        merged_track_id += 1

    merges_performed = len(valid_track_points) - len(merged_tracks)
    return merged_tracks, original_to_merged, merges_performed


def write_merged_tracking_data(
    output_file,
    merged_track_points,
    display_names,
):
    """Write ROI-filtered valid observations with original and merged IDs."""
    columns = [
        "merged_track_id",
        "original_track_id",
        "frame_index",
        "time_seconds",
        "class_id",
        "display_species_name",
        "scientific_species_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "box_width",
        "box_height",
        "zone",
    ]
    rows = []
    for merged_track_id, points in merged_track_points.items():
        for point in points:
            class_id = point["class_id"]
            rows.append(
                {
                    "merged_track_id": merged_track_id,
                    "original_track_id": point["original_track_id"],
                    "frame_index": point["frame_index"],
                    "time_seconds": point["time_seconds"],
                    "class_id": class_id,
                    "display_species_name": display_names[class_id],
                    "scientific_species_name": SCIENTIFIC_NAMES[class_id],
                    "confidence": point["confidence"],
                    "x1": point["x1"],
                    "y1": point["y1"],
                    "x2": point["x2"],
                    "y2": point["y2"],
                    "center_x": point["center_x"],
                    "center_y": point["center_y"],
                    "box_width": point["box_width"],
                    "box_height": point["box_height"],
                    "zone": point["zone"],
                }
            )
    rows.sort(key=lambda row: (row["frame_index"], row["merged_track_id"]))
    write_csv_rows(output_file, columns, rows)


def build_per_second_counts_from_tracks(track_points, final_species_by_track):
    """Build per-second species/zone sets using only the selected tracked IDs."""
    per_second_species = defaultdict(lambda: defaultdict(set))
    per_second_zones = defaultdict(lambda: defaultdict(set))
    for track_id, points in track_points.items():
        final_species_id = final_species_by_track[track_id]
        identity = ("track", track_id)
        for point in points:
            second = int(point["time_seconds"])
            per_second_species[second][final_species_id].add(identity)
            per_second_zones[second][point["zone"]].add(identity)
    return per_second_species, per_second_zones


def write_csv_rows(output_file, columns, rows):
    """Write dictionaries to one UTF-8 CSV with stable column order."""
    with output_file.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            formatted_row = {}
            for column in columns:
                value = row[column]
                formatted_row[column] = f"{value:.6f}" if isinstance(value, float) else value
            writer.writerow(formatted_row)


def write_species_counts(output_file, per_second_species, processed_seconds):
    """Write unique fish identities observed for every whole video second."""
    columns = ["time_seconds"] + [COMMON_NAMES[class_id] for class_id in range(8)] + [
        "total_fish"
    ]
    last_second = max(0, math.ceil(processed_seconds) - 1)
    rows = []
    for second in range(last_second + 1):
        row = {"time_seconds": second}
        all_identities = set()
        for class_id in range(8):
            identities = per_second_species[second][class_id]
            row[COMMON_NAMES[class_id]] = len(identities)
            all_identities.update(identities)
        row["total_fish"] = len(all_identities)
        rows.append(row)
    write_csv_rows(output_file, columns, rows)


def write_zone_counts(output_file, per_second_zones, processed_seconds):
    """Write unique fish identities observed in each zone for every second."""
    columns = ["time_seconds", "left", "center", "right", "total_fish"]
    last_second = max(0, math.ceil(processed_seconds) - 1)
    rows = []
    for second in range(last_second + 1):
        row = {"time_seconds": second}
        all_identities = set()
        for zone in ("left", "center", "right"):
            identities = per_second_zones[second][zone]
            row[zone] = len(identities)
            all_identities.update(identities)
        row["total_fish"] = len(all_identities)
        rows.append(row)
    write_csv_rows(output_file, columns, rows)


def prepare_background(representative_frame, frame_width, frame_height):
    """Return a dimmed RGB representative frame for research figures."""
    if representative_frame is None:
        representative_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    rgb_frame = cv2.cvtColor(representative_frame, cv2.COLOR_BGR2RGB)
    return (rgb_frame.astype(np.float32) * 0.58).astype(np.uint8)


def save_movement_heatmap(output_file, track_points, background, heatmap_alpha):
    """Overlay a 2D center-point density heatmap on the representative frame."""
    height, width = background.shape[:2]
    x_points = []
    y_points = []
    for points in track_points.values():
        x_points.extend(point["center_x"] for point in points)
        y_points.extend(point["center_y"] for point in points)

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(background)
    if x_points:
        heatmap, x_edges, y_edges = np.histogram2d(
            x_points,
            y_points,
            bins=(80, 45),
            range=((0, width), (0, height)),
        )
        axis.imshow(
            heatmap.T,
            extent=(x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]),
            cmap="inferno",
            alpha=heatmap_alpha,
            interpolation="bilinear",
        )
    else:
        axis.text(width / 2, height / 2, "No tracked points", color="white", ha="center")
    axis.set_title("Fish Movement Heatmap")
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def moving_average(values, window_size=5):
    """Smooth a short numeric sequence without shortening it."""
    if len(values) < window_size:
        return np.asarray(values, dtype=float)
    padding = window_size // 2
    padded = np.pad(np.asarray(values, dtype=float), padding, mode="edge")
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(padded, kernel, mode="valid")


def save_trajectory_overlay(
    output_file,
    track_points,
    final_species_by_track,
    background,
    trajectory_labels,
    trajectory_min_points,
    line_thickness,
    smooth_trajectories,
    max_trajectory_labels,
    merged_id_labels=False,
):
    """Draw each tracked trajectory over the representative video frame."""
    height, width = background.shape[:2]
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.imshow(background)
    color_map = plt.get_cmap("tab20")
    plotted_any = False
    eligible_track_ids = [
        track_id
        for track_id, points in track_points.items()
        if len(points) >= trajectory_min_points
    ]
    if trajectory_labels == "none":
        labeled_track_ids = set()
    else:
        longest_track_ids = sorted(
            eligible_track_ids,
            key=lambda track_id: (-len(track_points[track_id]), track_id),
        )
        labeled_track_ids = set(longest_track_ids[:max_trajectory_labels])

    for index, track_id in enumerate(sorted(track_points)):
        points = track_points[track_id]
        if len(points) < trajectory_min_points:
            continue
        x_values = [point["center_x"] for point in points]
        y_values = [point["center_y"] for point in points]
        if smooth_trajectories:
            x_values = moving_average(x_values, window_size=5)
            y_values = moving_average(y_values, window_size=5)
        color = color_map(index % 20)
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=line_thickness,
            alpha=0.82,
        )
        plotted_any = True
        if len(x_values) and track_id in labeled_track_ids:
            species_id = final_species_by_track[track_id]
            label = (
                (f"M{track_id}" if merged_id_labels else f"ID{track_id}")
                if trajectory_labels == "id"
                else COMMON_NAMES[species_id]
            )
            axis.text(
                x_values[-1],
                y_values[-1],
                label,
                color="white",
                fontsize=6.5,
                bbox={"facecolor": color, "alpha": 0.70, "pad": 2},
            )

    if not plotted_any:
        axis.text(width / 2, height / 2, "No tracked trajectories", color="white", ha="center")
    axis.set_title("Tracked Fish Trajectories")
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_species_duration_bar(output_file, summaries):
    """Plot total smoothed track duration for every species."""
    durations = Counter()
    for summary in summaries:
        durations[summary["final_species_id"]] += summary["duration_seconds"]
    sorted_species = sorted(
        range(8),
        key=lambda class_id: (-durations[class_id], COMMON_NAMES[class_id]),
    )
    names = [COMMON_NAMES[class_id] for class_id in sorted_species]
    values = [durations[class_id] for class_id in sorted_species]

    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(names, values, color="#2e86ab")
    axis.invert_yaxis()
    axis.set_title("Cumulative Tracked Duration by Species")
    axis.set_xlabel("Cumulative tracklet-seconds")
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4)
    maximum_value = max(values, default=0.0)
    axis.set_xlim(0, maximum_value * 1.14 + 0.5)
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def save_zone_occupancy_bar(output_file, zone_rows):
    """Plot total observed time in the left, center, and right zones."""
    values = [
        sum(row[f"{zone}_time_seconds"] for row in zone_rows)
        for zone in ("left", "center", "right")
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        ("Left", "Center", "Right"),
        values,
        color=("#4c78a8", "#f2cf5b", "#e45756"),
    )
    axis.set_title("Cumulative Zone Occupancy")
    axis.set_ylabel("Cumulative tracklet-seconds")
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3)
    maximum_value = max(values, default=0.0)
    axis.set_ylim(0, maximum_value * 1.14 + 0.5)
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def save_tracklet_merge_summary(output_file, raw_count, valid_count, merged_count):
    """Plot raw, quality-filtered, and conservatively merged track counts."""
    labels = ("Raw tracks", "Valid tracks", "Merged tracks")
    values = (raw_count, valid_count, merged_count)
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, values, color=("#9c9c9c", "#4c78a8", "#59a14f"))
    axis.set_title("Tracklet Merge Summary")
    axis.set_ylabel("Number of tracks")
    axis.bar_label(bars, padding=3)
    axis.set_ylim(0, max(values, default=0) * 1.18 + 1)
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)


def write_analysis_summary(
    output_file,
    args,
    processed_frames,
    source_fps,
    processed_seconds,
    frame_width,
    frame_height,
    raw_track_count,
    valid_track_count,
    merged_track_count,
    merges_performed,
    removed_track_count,
):
    """Write key inputs, thresholds, and track counts for reproducibility."""
    lines = [
        "Aquarium Fish Tracking Analysis Summary",
        "=======================================",
        f"Source video: {args.source}",
        f"Processed frames: {processed_frames}",
        f"Video FPS: {source_fps:.6f}",
        f"Processed video duration seconds: {processed_seconds:.6f}",
        f"Source frame size: {frame_width}x{frame_height}",
        f"Raw track count: {raw_track_count}",
        f"Valid track count: {valid_track_count}",
    ]
    if args.merge_tracklets:
        lines.append(f"Merged track count: {merged_track_count}")
    lines.extend(
        [
            f"Number of merges: {merges_performed}",
            f"Removed short/noisy tracks: {removed_track_count}",
            f"Tracker used: {args.tracker}",
            f"Confidence threshold: {args.conf}",
            f"YOLO image size: {args.imgsz}",
            f"Minimum track frames: {args.min_track_frames}",
            f"Minimum track duration seconds: {args.min_track_duration}",
        ]
    )
    if args.analysis_roi is not None:
        lines.append(f"Analysis ROI: {args.analysis_roi}")
    if args.merge_tracklets:
        lines.extend(
            [
                f"Merge max gap seconds: {args.merge_max_gap_seconds}",
                f"Merge max distance pixels: {args.merge_max_distance_pixels}",
                f"Merge same species only: {args.merge_same_species_only}",
                f"Merge minimum IoU: {args.merge_min_iou}",
            ]
        )
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_video(args):
    """Run tracking, stream frame rows to CSV, and generate all outputs."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "tracking": args.output_dir / "tracking_data.csv",
        "movement": args.output_dir / "movement_summary.csv",
        "species_counts": args.output_dir / "species_counts_over_time.csv",
        "zone_occupancy": args.output_dir / "zone_occupancy.csv",
        "zone_counts": args.output_dir / "zone_counts_over_time.csv",
        "heatmap": args.output_dir / "movement_heatmap.png",
        "trajectories": args.output_dir / "trajectory_overlay.png",
        "species_duration": args.output_dir / "species_duration_bar.png",
        "zone_bar": args.output_dir / "zone_occupancy_bar.png",
        "merged_tracking": args.output_dir / "merged_tracking_data.csv",
        "merged_movement": args.output_dir / "merged_movement_summary.csv",
        "merged_zone_occupancy": args.output_dir / "merged_zone_occupancy.csv",
        "movement_unmerged": args.output_dir / "movement_summary_unmerged.csv",
        "trajectory_unmerged": args.output_dir / "trajectory_overlay_unmerged.png",
        "trajectory_clean": args.output_dir / "trajectory_overlay_clean.png",
        "merge_summary": args.output_dir / "tracklet_merge_summary.png",
        "analysis_summary": args.output_dir / "analysis_summary.txt",
        "video": args.output_dir / "annotated_tracking_video.mp4",
    }

    capture, source_fps, frame_width, frame_height, total_frames = open_source_video(
        args.source
    )
    writer = None
    tracking_file = None
    frame_index = 0
    start_time = time.perf_counter()
    representative_frame = None
    track_points = defaultdict(list)
    per_second_species = defaultdict(lambda: defaultdict(set))
    per_second_zones = defaultdict(lambda: defaultdict(set))
    display_names = CHINESE_NAMES if args.label_lang == "zh" else COMMON_NAMES
    video_label_lang = "en" if args.clean_video else args.label_lang
    chinese_font = load_chinese_font(frame_width) if video_label_lang == "zh" else None

    target_frames = total_frames if total_frames > 0 else None
    if args.max_frames is not None:
        target_frames = (
            min(total_frames, args.max_frames) if total_frames > 0 else args.max_frames
        )
    representative_index = target_frames // 2 if target_frames else 0

    try:
        if args.save_video:
            writer_fps = args.output_fps if args.output_fps is not None else source_fps
            writer = create_video_writer(
                paths["video"],
                writer_fps,
                frame_width,
                frame_height,
            )

        tracking_file = paths["tracking"].open(
            "w",
            newline="",
            encoding="utf-8-sig",
        )
        tracking_writer = csv.DictWriter(tracking_file, fieldnames=TRACKING_COLUMNS)
        tracking_writer.writeheader()

        print(f"Loading model: {args.model}")
        model = YOLO(str(args.model))
        print(f"Source: {args.source}")
        print(f"Frame size: {frame_width}x{frame_height} | FPS: {source_fps:.3f}")
        print(f"Tracker: {args.tracker}")
        if args.analysis_roi is not None:
            print(f"Active analysis ROI: {args.analysis_roi}")

        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            success, frame = capture.read()
            if not success:
                break

            if representative_frame is None or frame_index == representative_index:
                representative_frame = frame.copy()

            time_seconds = frame_index / source_fps
            results = model.track(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                persist=True,
                tracker=args.tracker,
                verbose=False,
            )
            result = results[0]
            detections = extract_detections(
                result,
                frame_width,
                frame_height,
                args.hide_classes,
            )

            for detection in detections:
                write_tracking_row(
                    tracking_writer,
                    detection,
                    frame_index,
                    time_seconds,
                    display_names,
                )

            analysis_detections = [
                detection
                for detection in detections
                if center_is_inside_analysis_roi(
                    detection,
                    args.analysis_roi,
                    frame_width,
                    frame_height,
                )
            ]
            for detection in analysis_detections:
                second = int(time_seconds)
                update_per_second_counts(
                    per_second_species,
                    per_second_zones,
                    second,
                    detection,
                    frame_index,
                )

                if detection["track_id"] is not None:
                    track_points[detection["track_id"]].append(
                        {
                            "frame_index": frame_index,
                            "original_track_id": detection["track_id"],
                            "time_seconds": time_seconds,
                            "class_id": detection["class_id"],
                            "confidence": detection["confidence"],
                            "center_x": detection["center_x"],
                            "center_y": detection["center_y"],
                            "box_area": detection["box_area"],
                            "x1": detection["x1"],
                            "y1": detection["y1"],
                            "x2": detection["x2"],
                            "y2": detection["y2"],
                            "box_width": detection["box_width"],
                            "box_height": detection["box_height"],
                            "zone": detection["zone"],
                        }
                    )

            if writer is not None:
                if video_label_lang == "zh":
                    annotated_frame = draw_chinese_overlay(
                        frame,
                        detections,
                        chinese_font,
                        args,
                    )
                else:
                    annotated_frame = draw_english_overlay(frame, detections, args)
                writer.write(annotated_frame)

            frame_index += 1
            if frame_index % 100 == 0:
                elapsed = time.perf_counter() - start_time
                processing_fps = frame_index / elapsed if elapsed > 0 else 0.0
                print(
                    f"Processed frame {frame_index} | elapsed {elapsed:.1f}s | "
                    f"processing FPS {processing_fps:.2f}"
                )

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if tracking_file is not None:
            tracking_file.close()

    processed_seconds = frame_index / source_fps
    raw_track_count = len(track_points)
    valid_track_points = filter_valid_tracks(
        track_points,
        args.min_track_frames,
        args.min_track_duration,
    )
    valid_track_count = len(valid_track_points)
    removed_track_count = raw_track_count - valid_track_count
    unmerged_summaries, unmerged_zone_rows, unmerged_species_by_track = (
        build_track_summaries(
            valid_track_points,
            source_fps,
        )
    )
    merged_track_points, original_to_merged, merges_performed = (
        merge_valid_tracklets(
            valid_track_points,
            unmerged_species_by_track,
            args,
        )
    )
    merged_track_count = len(merged_track_points)
    merged_summaries, merged_zone_rows, merged_species_by_track = (
        build_track_summaries(
            merged_track_points,
            source_fps,
        )
    )

    if args.merge_tracklets:
        analysis_track_points = merged_track_points
        summaries = merged_summaries
        zone_rows = merged_zone_rows
        final_species_by_track = merged_species_by_track
    else:
        analysis_track_points = valid_track_points
        summaries = unmerged_summaries
        zone_rows = unmerged_zone_rows
        final_species_by_track = unmerged_species_by_track

    movement_columns = [
        "track_id",
        "final_species_id",
        "final_species_common_name",
        "final_species_scientific_name",
        "first_seen_time",
        "last_seen_time",
        "duration_seconds",
        "num_frames_seen",
        "mean_confidence",
        "total_path_length_pixels",
        "average_speed_pixels_per_second",
        "max_speed_pixels_per_second",
        "mean_box_area",
        "zone_most_used",
    ]
    zone_columns = [
        "track_id",
        "species",
        "left_time_seconds",
        "center_time_seconds",
        "right_time_seconds",
        "most_used_zone",
    ]

    write_merged_tracking_data(
        paths["merged_tracking"],
        merged_track_points,
        display_names,
    )
    write_csv_rows(paths["merged_movement"], movement_columns, merged_summaries)
    write_csv_rows(
        paths["merged_zone_occupancy"],
        zone_columns,
        merged_zone_rows,
    )
    write_csv_rows(paths["movement"], movement_columns, summaries)
    write_csv_rows(paths["zone_occupancy"], zone_columns, zone_rows)
    if args.merge_tracklets:
        write_csv_rows(
            paths["movement_unmerged"],
            movement_columns,
            unmerged_summaries,
        )

    if args.filter_summary_only:
        counts_species = per_second_species
        counts_zones = per_second_zones
    else:
        # Per-second counts continue to use quality-filtered original tracklets.
        # Merging changes identity continuity, not the fish visible in a second.
        counts_species, counts_zones = build_per_second_counts_from_tracks(
            valid_track_points,
            unmerged_species_by_track,
        )
    write_species_counts(paths["species_counts"], counts_species, processed_seconds)
    write_zone_counts(paths["zone_counts"], counts_zones, processed_seconds)

    background = prepare_background(
        representative_frame,
        frame_width,
        frame_height,
    )
    save_movement_heatmap(
        paths["heatmap"],
        analysis_track_points,
        background,
        args.heatmap_alpha,
    )
    save_trajectory_overlay(
        paths["trajectories"],
        analysis_track_points,
        final_species_by_track,
        background,
        args.trajectory_labels,
        args.trajectory_min_points,
        args.trajectory_line_thickness,
        args.smooth_trajectories,
        args.max_trajectory_labels,
        merged_id_labels=args.merge_tracklets,
    )
    if args.merge_tracklets:
        save_trajectory_overlay(
            paths["trajectory_unmerged"],
            valid_track_points,
            unmerged_species_by_track,
            background,
            args.trajectory_labels,
            args.trajectory_min_points,
            args.trajectory_line_thickness,
            args.smooth_trajectories,
            args.max_trajectory_labels,
            merged_id_labels=False,
        )
    save_trajectory_overlay(
        paths["trajectory_clean"],
        analysis_track_points,
        final_species_by_track,
        background,
        "none",
        args.trajectory_min_points,
        args.trajectory_line_thickness,
        args.smooth_trajectories,
        0,
        merged_id_labels=args.merge_tracklets,
    )
    save_species_duration_bar(paths["species_duration"], summaries)
    save_zone_occupancy_bar(paths["zone_bar"], zone_rows)
    save_tracklet_merge_summary(
        paths["merge_summary"],
        raw_track_count,
        valid_track_count,
        merged_track_count,
    )
    write_analysis_summary(
        paths["analysis_summary"],
        args,
        frame_index,
        source_fps,
        processed_seconds,
        frame_width,
        frame_height,
        raw_track_count,
        valid_track_count,
        merged_track_count,
        merges_performed,
        removed_track_count,
    )

    elapsed = time.perf_counter() - start_time
    print("\nAnalysis complete")
    print(f"Processed frames: {frame_index}")
    print(f"Raw track count: {raw_track_count}")
    print(f"Valid track count before merging: {valid_track_count}")
    print(f"Merged track count: {merged_track_count}")
    print(f"Number of merges performed: {merges_performed}")
    print(f"Removed short/noisy track count: {removed_track_count}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print("Generated files:")
    for name, output_path in paths.items():
        if name == "video" and not args.save_video:
            continue
        if name in ("movement_unmerged", "trajectory_unmerged") and not args.merge_tracklets:
            continue
        print(f"  {output_path}")


def main():
    args = parse_arguments()
    try:
        validate_arguments(args)
        analyze_video(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
