from pathlib import Path
import argparse
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Defaults and display names
# ---------------------------------------------------------------------------
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "weights" / "reef_ai_yolov12n_v2_full_tank.pt"
CHINESE_FONT_FILE = Path(r"C:\Windows\Fonts\msjh.ttc")
PREVIEW_WINDOW_NAME = "Aquarium YOLO Demo Render"

CHINESE_LABELS = {
    0: "藍倒吊",
    1: "雷達魚",
    2: "橘小丑",
    3: "黑小丑",
    4: "黑白雀",
    5: "黃雀鯛",
    6: "青魔",
    7: "蝙蝠魚",
}

ENGLISH_LABELS = {
    0: "Blue tang",
    1: "Firefish",
    2: "Orange clown",
    3: "Black clown",
    4: "Humbug damsel",
    5: "Lemon damsel",
    6: "Green chromis",
    7: "Silver moony",
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


def parse_arguments():
    """Read command-line settings for the offline render."""
    parser = argparse.ArgumentParser(
        description="Render a polished aquarium YOLO demo video."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Input video path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output MP4 path",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="YOLO model path",
    )
    parser.add_argument("--track", action="store_true", help="Use ByteTrack")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Minimum detection confidence (default: 0.45)",
    )
    parser.add_argument(
        "--hide_id",
        action="store_true",
        help="Hide tracking IDs while tracking remains active",
    )
    parser.add_argument(
        "--hide_conf",
        action="store_true",
        help="Hide confidence values",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current-frame species counts",
    )
    parser.add_argument(
        "--box_thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness (default: 2)",
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=30.0,
        help="Output video playback FPS (default: 30)",
    )
    parser.add_argument(
        "--start_seconds",
        type=float,
        default=0.0,
        help="Start position in the input video (default: 0)",
    )
    parser.add_argument(
        "--duration_seconds",
        type=float,
        default=None,
        help="Optional clip duration in seconds",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a preview window while rendering",
    )
    return parser.parse_args()


def validate_arguments(args):
    """Check paths and numeric options before loading the model."""
    if not args.source.is_file():
        raise FileNotFoundError(f"Input video was not found: {args.source}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model was not found: {args.model}")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("--source and --output must be different files")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if args.box_thickness <= 0:
        raise ValueError("--box_thickness must be greater than 0")
    if args.output_fps <= 0:
        raise ValueError("--output_fps must be greater than 0")
    if args.start_seconds < 0:
        raise ValueError("--start_seconds cannot be negative")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("--duration_seconds must be greater than 0")


def load_label_font(frame_width):
    """Load Microsoft JhengHei for Chinese, with an English fallback."""
    font_size = max(12, round(18 * frame_width / 1280))
    if CHINESE_FONT_FILE.is_file():
        try:
            return ImageFont.truetype(str(CHINESE_FONT_FILE), font_size), True
        except OSError as error:
            print(f"WARNING: Could not load {CHINESE_FONT_FILE}: {error}")

    print(
        f"WARNING: Chinese font not available at {CHINESE_FONT_FILE}; "
        "using English labels."
    )
    return ImageFont.load_default(), False


def run_inference(model, frame, args):
    """Run either ordinary prediction or persistent ByteTrack tracking."""
    common_options = {
        "source": frame,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
    }
    if args.track:
        return model.track(
            **common_options,
            persist=True,
            tracker="bytetrack.yaml",
        )
    return model.predict(**common_options)


def make_label_text(box, class_id, confidence, labels, args):
    """Build a compact label according to ID and confidence options."""
    parts = []
    if args.track and not args.hide_id and box.id is not None:
        tracking_id = int(box.id[0].item())
        parts.append(f"ID{tracking_id}")

    parts.append(labels.get(class_id, str(class_id)))
    if not args.hide_conf:
        parts.append(f"{confidence:.2f}")
    return " ".join(parts)


def draw_detection_overlay(frame, result, font, use_chinese, args):
    """Draw thin boxes and compact Pillow-rendered labels."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    labels = CHINESE_LABELS if use_chinese else ENGLISH_LABELS
    frame_height, frame_width = frame.shape[:2]
    label_items = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        x1 = max(0, min(frame_width - 1, round(x1)))
        y1 = max(0, min(frame_height - 1, round(y1)))
        x2 = max(0, min(frame_width - 1, round(x2)))
        y2 = max(0, min(frame_height - 1, round(y2)))

        color = BOX_COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            args.box_thickness,
        )
        label_text = make_label_text(box, class_id, confidence, labels, args)
        label_items.append((x1, y1, label_text, color))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(rgb_frame)
    image_draw = ImageDraw.Draw(pillow_image)

    for x1, y1, label_text, bgr_color in label_items:
        text_box = image_draw.textbbox((0, 0), label_text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        padding_x = 3
        padding_y = 2
        background_width = text_width + padding_x * 2
        background_height = text_height + padding_y * 2

        label_x = min(x1, max(0, frame_width - background_width))
        label_y = y1 - background_height - 2
        if label_y < 0:
            label_y = min(frame_height - background_height, y1 + 2)

        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        image_draw.rectangle(
            (
                label_x,
                label_y,
                label_x + background_width,
                label_y + background_height,
            ),
            fill=(0, 0, 0),
            outline=rgb_color,
            width=1,
        )
        image_draw.text(
            (label_x + padding_x, label_y + padding_y - text_box[1]),
            label_text,
            font=font,
            fill=rgb_color,
        )

    return cv2.cvtColor(np.asarray(pillow_image), cv2.COLOR_RGB2BGR)


def draw_statistics_panel(frame, result, font, use_chinese):
    """Draw a compact semi-transparent panel for classes in this frame."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    class_counts = {}
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    labels = CHINESE_LABELS if use_chinese else ENGLISH_LABELS
    title = "目前偵測" if use_chinese else "Detected now"
    lines = [title]
    for class_id in sorted(class_counts):
        lines.append(f"{labels.get(class_id, str(class_id))} x{class_counts[class_id]}")

    base_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert(
        "RGBA"
    )
    measurement_draw = ImageDraw.Draw(base_image)
    padding = 8
    line_gap = 3
    line_sizes = []
    for line in lines:
        bounds = measurement_draw.textbbox((0, 0), line, font=font)
        line_sizes.append((bounds[2] - bounds[0], bounds[3] - bounds[1], bounds[1]))

    panel_width = max(width for width, _height, _top in line_sizes) + padding * 2
    panel_height = (
        sum(height for _width, height, _top in line_sizes)
        + line_gap * (len(lines) - 1)
        + padding * 2
    )
    frame_width, frame_height = base_image.size
    panel_x = max(0, frame_width - panel_width - 10)
    panel_y = 10 if panel_height <= frame_height - 20 else 0

    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
        radius=5,
        fill=(0, 0, 0, 175),
        outline=(215, 215, 215, 190),
        width=1,
    )

    text_y = panel_y + padding
    for index, (line, (_width, line_height, text_top)) in enumerate(
        zip(lines, line_sizes)
    ):
        color = (255, 230, 120, 255) if index == 0 else (255, 255, 255, 255)
        overlay_draw.text(
            (panel_x + padding, text_y - text_top),
            line,
            font=font,
            fill=color,
        )
        text_y += line_height + line_gap

    combined_image = Image.alpha_composite(base_image, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(combined_image), cv2.COLOR_RGB2BGR)


def open_video_capture(source, start_seconds):
    """Open the input video and seek to the requested starting position."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid FPS or frame dimensions")

    start_source_frame = round(start_seconds * source_fps)
    if total_source_frames > 0 and start_source_frame >= total_source_frames:
        capture.release()
        raise ValueError("--start_seconds is beyond the end of the input video")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_source_frame)
    return (
        capture,
        source_fps,
        frame_width,
        frame_height,
        total_source_frames,
        start_source_frame,
    )


def create_video_writer(output_file, output_fps, frame_width, frame_height):
    """Create an MP4 writer using the original source frame dimensions."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_file),
        fourcc,
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_file}")
    return writer


def read_until_source_frame(capture, current_index, target_index):
    """Read sequentially until reaching the source frame needed for output."""
    frame = None
    while current_index <= target_index:
        success, frame = capture.read()
        if not success:
            return None, current_index
        current_index += 1
    return frame, current_index


def render_video(args):
    """Render the selected clip with YOLO annotations."""
    (
        capture,
        source_fps,
        frame_width,
        frame_height,
        total_source_frames,
        start_source_frame,
    ) = open_video_capture(args.source, args.start_seconds)

    writer = None
    rendered_frame_count = 0
    current_source_index = start_source_frame
    last_target_source_index = None
    last_annotated_frame = None
    start_time = time.perf_counter()

    if args.duration_seconds is not None:
        maximum_output_frames = round(args.duration_seconds * args.output_fps)
    elif total_source_frames > 0:
        remaining_seconds = (total_source_frames - start_source_frame) / source_fps
        maximum_output_frames = round(remaining_seconds * args.output_fps)
    else:
        maximum_output_frames = None

    try:
        writer = create_video_writer(
            args.output,
            args.output_fps,
            frame_width,
            frame_height,
        )
        font, use_chinese = load_label_font(frame_width)
        print(f"Loading model: {args.model}")
        model = YOLO(str(args.model))
        print(f"Source: {args.source}")
        print(f"Output: {args.output}")
        print(
            f"Frame size: {frame_width}x{frame_height} | "
            f"source FPS: {source_fps:.3f} | output FPS: {args.output_fps:.3f}"
        )

        while maximum_output_frames is None or rendered_frame_count < maximum_output_frames:
            clip_time = rendered_frame_count / args.output_fps
            target_source_index = start_source_frame + int(clip_time * source_fps)

            if target_source_index != last_target_source_index:
                source_frame, current_source_index = read_until_source_frame(
                    capture,
                    current_source_index,
                    target_source_index,
                )
                if source_frame is None:
                    break

                results = run_inference(model, source_frame, args)
                annotated_frame = draw_detection_overlay(
                    source_frame,
                    results[0],
                    font,
                    use_chinese,
                    args,
                )
                if args.stats:
                    annotated_frame = draw_statistics_panel(
                        annotated_frame,
                        results[0],
                        font,
                        use_chinese,
                    )

                last_annotated_frame = annotated_frame
                last_target_source_index = target_source_index

            # When output FPS is higher than source FPS, the most recent
            # annotated frame is repeated. This preserves duration and creates
            # a valid, smoothly timed output stream at --output_fps.
            if last_annotated_frame is None:
                break
            writer.write(last_annotated_frame)
            rendered_frame_count += 1

            if rendered_frame_count % 30 == 0:
                elapsed = time.perf_counter() - start_time
                processing_fps = rendered_frame_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"Processed frame {rendered_frame_count} | "
                    f"elapsed {elapsed:.1f}s | processing FPS {processing_fps:.2f}"
                )

            if args.show:
                cv2.imshow(PREVIEW_WINDOW_NAME, last_annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview stopped by user.")
                    break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    print("\nRender complete")
    print(f"Frames written: {rendered_frame_count}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Output video: {args.output}")


def main():
    args = parse_arguments()
    try:
        validate_arguments(args)
        render_video(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
