from pathlib import Path
import argparse
from datetime import datetime
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
DEFAULT_URL = None
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "weights" / "reef_ai_yolov12n_v2_full_tank.pt"
FFMPEG_EXE = "ffmpeg"
FFPROBE_EXE = "ffprobe"
SCREENSHOT_FOLDER = Path(__file__).resolve().parents[1] / "outputs" / "live_screenshots"
WINDOW_NAME = "Aquarium YOLO Live"
CHINESE_FONT_FILE = Path(r"C:\Windows\Fonts\msjh.ttc")

# Short display names. The class IDs must match the training dataset order.
SHORT_LABELS = {
    0: "藍倒吊",
    1: "雷達魚",
    2: "橘小丑",
    3: "黑小丑",
    4: "黑白雀",
    5: "黃雀鯛",
    6: "青魔",
    7: "蝙蝠魚",
}

ENGLISH_SHORT_LABELS = {
    0: "Blue tang",
    1: "Firefish",
    2: "Orange clown",
    3: "Black clown",
    4: "Humbug damsel",
    5: "Lemon damsel",
    6: "Green chromis",
    7: "Silver moony",
}

# A small, distinct BGR color for each class.
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
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Run a YOLO model on a YouTube Live stream through FFmpeg."
    )
    parser.add_argument(
        "--source_type",
        choices=("youtube", "video", "camera"),
        default="youtube",
        help="Input type: youtube, video, or camera (default: youtube)",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="YouTube Live URL")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Local video path, used only with --source_type video",
    )
    parser.add_argument(
        "--loop_video",
        action="store_true",
        help="Restart a local video from frame 0 when it ends",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="USB camera device ID (default: 0)",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=None,
        help="Optional requested USB camera capture width",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=None,
        help="Optional requested USB camera capture height",
    )
    parser.add_argument(
        "--camera_fps",
        type=float,
        default=None,
        help="Optional requested USB camera capture FPS",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the YOLO .pt model",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width used for YOLO processing (default: 1280)",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=None,
        help="Optional final display width; does not change YOLO processing size",
    )
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
        "--roi",
        type=parse_roi,
        default=None,
        metavar="X1,Y1,X2,Y2",
        help=(
            "Normalized detection region, for example "
            "--roi 0.00,0.34,0.55,0.78"
        ),
    )
    parser.add_argument(
        "--draw_roi",
        action="store_true",
        help="Draw a thin boundary around the detection ROI",
    )
    parser.add_argument(
        "--hide_conf",
        action="store_true",
        help="Hide confidence values from detection labels",
    )
    parser.add_argument(
        "--hide_id",
        action="store_true",
        help="Hide tracking IDs while keeping tracking active",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Use persistent BoT-SORT tracking instead of predict mode",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config used with --track (default: bytetrack.yaml)",
    )
    parser.add_argument(
        "--process_every",
        type=int,
        default=1,
        help="Run YOLO every N frames and reuse detections between them (default: 1)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current-frame fish counts by species",
    )
    parser.add_argument(
        "--box_thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness (default: 2)",
    )
    return parser.parse_args()


def parse_roi(value):
    """Convert X1,Y1,X2,Y2 text into four validated normalized values."""
    try:
        coordinates = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ROI must contain four numbers: x1,y1,x2,y2"
        ) from error

    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError(
            "ROI must contain exactly four values: x1,y1,x2,y2"
        )

    x1, y1, x2, y2 = coordinates
    if any(coordinate < 0.0 or coordinate > 1.0 for coordinate in coordinates):
        raise argparse.ArgumentTypeError("ROI values must be between 0 and 1")
    if x1 >= x2 or y1 >= y2:
        raise argparse.ArgumentTypeError("ROI requires x1 < x2 and y1 < y2")

    return coordinates


def check_settings(args):
    """Check local paths and numeric options before starting subprocesses."""
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file was not found: {args.model}")
    if args.source_type == "youtube":
        if not args.url:
            raise ValueError("--url is required when --source_type youtube is used")
        if not shutil.which(FFMPEG_EXE):
            raise FileNotFoundError("FFmpeg was not found on PATH")
        if not shutil.which(FFPROBE_EXE):
            raise FileNotFoundError("ffprobe was not found on PATH")
    if args.source_type == "video":
        if args.video is None:
            raise ValueError("--video is required when --source_type video is used")
        if not args.video.is_file():
            raise FileNotFoundError(f"Local video was not found: {args.video}")
    if args.camera < 0:
        raise ValueError("--camera cannot be negative")
    if args.camera_width is not None and args.camera_width <= 0:
        raise ValueError("--camera_width must be greater than 0")
    if args.camera_height is not None and args.camera_height <= 0:
        raise ValueError("--camera_height must be greater than 0")
    if args.camera_fps is not None and args.camera_fps <= 0:
        raise ValueError("--camera_fps must be greater than 0")
    if args.width <= 0:
        raise ValueError("--width must be greater than 0")
    if args.display_width is not None and args.display_width <= 0:
        raise ValueError("--display_width must be greater than 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if args.box_thickness <= 0:
        raise ValueError("--box_thickness must be greater than 0")
    if args.process_every <= 0:
        raise ValueError("--process_every must be greater than 0")
    if not args.tracker.strip():
        raise ValueError("--tracker cannot be empty")


def get_m3u8_url(youtube_url):
    """Ask yt-dlp for the direct HLS/m3u8 stream URL."""
    command = [
        "yt-dlp",
        "-f",
        "best[protocol*=m3u8]/best",
        "-g",
        youtube_url,
    ]
    print("Resolving the YouTube Live stream with yt-dlp...")
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    direct_urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not direct_urls:
        raise RuntimeError("yt-dlp did not return a direct stream URL")

    # This format normally returns one combined HLS stream. If yt-dlp prints
    # more than one URL, the first is the video stream needed by FFmpeg.
    return direct_urls[0]


def get_stream_size(m3u8_url, output_width):
    """Read source dimensions with ffprobe and calculate scale=WIDTH:-2 height."""
    command = [
        str(FFPROBE_EXE),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        m3u8_url,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    size_line = next(
        (line.strip() for line in result.stdout.splitlines() if "x" in line),
        "",
    )
    try:
        source_width_text, source_height_text = size_line.split("x", maxsplit=1)
        source_width = int(source_width_text)
        source_height = int(source_height_text)
    except (ValueError, TypeError) as error:
        raise RuntimeError(
            f"Could not determine stream dimensions from ffprobe output: {size_line!r}"
        ) from error

    # FFmpeg's -2 scale value preserves aspect ratio and makes the result even.
    output_height = round((source_height * output_width / source_width) / 2) * 2
    if output_height <= 0:
        raise RuntimeError("Calculated stream height is invalid")

    return output_width, output_height


def start_ffmpeg(m3u8_url, output_width):
    """Start FFmpeg and return a process that emits raw BGR frames."""
    command = [
        str(FFMPEG_EXE),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        m3u8_url,
        "-an",
        "-vf",
        f"scale={output_width}:-2",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    print("Starting the FFmpeg video pipe...")
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        # Keep FFmpeg warnings visible without risking a filled stderr pipe.
        stderr=None,
        bufsize=10**8,
    )


def read_exactly(stream, byte_count):
    """Read one complete frame, even if the pipe returns smaller chunks."""
    frame_data = bytearray()
    while len(frame_data) < byte_count:
        chunk = stream.read(byte_count - len(frame_data))
        if not chunk:
            return None
        frame_data.extend(chunk)
    return bytes(frame_data)


def load_label_font(frame_width):
    """Load a Unicode font, or return an English-label fallback."""
    font_size = max(12, round(18 * frame_width / 1280))
    if CHINESE_FONT_FILE.is_file():
        try:
            return ImageFont.truetype(str(CHINESE_FONT_FILE), font_size), True
        except OSError as error:
            print(f"WARNING: Could not load {CHINESE_FONT_FILE}: {error}")

    print(
        f"WARNING: Chinese font not available at {CHINESE_FONT_FILE}; "
        "using English short labels."
    )
    return ImageFont.load_default(), False


def roi_pixel_bounds(frame, normalized_roi):
    """Convert a normalized ROI into safe full-frame pixel coordinates."""
    frame_height, frame_width = frame.shape[:2]
    if normalized_roi is None:
        return 0, 0, frame_width, frame_height

    x1_ratio, y1_ratio, x2_ratio, y2_ratio = normalized_roi
    x1 = max(0, min(frame_width - 1, round(x1_ratio * frame_width)))
    y1 = max(0, min(frame_height - 1, round(y1_ratio * frame_height)))
    x2 = max(x1 + 1, min(frame_width, round(x2_ratio * frame_width)))
    y2 = max(y1 + 1, min(frame_height, round(y2_ratio * frame_height)))
    return x1, y1, x2, y2


def make_label_text(
    box,
    class_id,
    confidence,
    use_chinese,
    hide_conf,
    hide_id,
):
    """Build a compact detection label, optionally including ID and confidence."""
    labels = SHORT_LABELS if use_chinese else ENGLISH_SHORT_LABELS
    parts = []

    if box.id is not None and not hide_id:
        tracking_id = int(box.id[0].item())
        parts.append(f"ID{tracking_id}")

    parts.append(labels.get(class_id, str(class_id)))
    if not hide_conf:
        parts.append(f"{confidence:.2f}")
    return " ".join(parts)


def draw_detections(
    frame,
    result,
    offset_x,
    offset_y,
    font,
    use_chinese,
    hide_conf,
    hide_id,
    box_thickness,
):
    """Draw thin full-frame boxes and compact Pillow text labels."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    label_items = []
    frame_height, frame_width = frame.shape[:2]

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        x1 = max(0, min(frame_width - 1, round(x1 + offset_x)))
        y1 = max(0, min(frame_height - 1, round(y1 + offset_y)))
        x2 = max(0, min(frame_width - 1, round(x2 + offset_x)))
        y2 = max(0, min(frame_height - 1, round(y2 + offset_y)))

        color = BOX_COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        label_text = make_label_text(
            box,
            class_id,
            confidence,
            use_chinese,
            hide_conf,
            hide_id,
        )
        label_items.append((x1, y1, label_text, color))

    # Convert once per frame and draw all Unicode labels with Pillow.
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


def draw_species_stats(frame, result, font, use_chinese):
    """Draw a compact, semi-transparent current-frame species count panel."""
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    class_counts = {}
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

    labels = SHORT_LABELS if use_chinese else ENGLISH_SHORT_LABELS
    title = "目前偵測" if use_chinese else "Detected now"
    panel_lines = [title]
    for class_id in sorted(class_counts):
        class_name = labels.get(class_id, str(class_id))
        panel_lines.append(f"{class_name} x{class_counts[class_id]}")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    base_image = Image.fromarray(rgb_frame).convert("RGBA")
    measurement_draw = ImageDraw.Draw(base_image)

    padding = 8
    line_gap = 3
    line_sizes = []
    for line in panel_lines:
        bounds = measurement_draw.textbbox((0, 0), line, font=font)
        line_sizes.append((bounds[2] - bounds[0], bounds[3] - bounds[1], bounds[1]))

    panel_width = max(width for width, _height, _top in line_sizes) + padding * 2
    panel_height = (
        sum(height for _width, height, _top in line_sizes)
        + line_gap * (len(panel_lines) - 1)
        + padding * 2
    )

    frame_width, frame_height = base_image.size
    panel_x = max(0, frame_width - panel_width - 10)
    panel_y = 10
    if panel_height > frame_height - 20:
        panel_y = 0

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
        zip(panel_lines, line_sizes)
    ):
        text_color = (255, 230, 120, 255) if index == 0 else (255, 255, 255, 255)
        overlay_draw.text(
            (panel_x + padding, text_y - text_top),
            line,
            font=font,
            fill=text_color,
        )
        text_y += line_height + line_gap

    combined_image = Image.alpha_composite(base_image, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(combined_image), cv2.COLOR_RGB2BGR)


def run_inference(model, inference_frame, args):
    """Run either ordinary prediction or persistent BoT-SORT tracking."""
    common_options = {
        "source": inference_frame,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
    }
    if args.track:
        return model.track(
            **common_options,
            persist=True,
            tracker=args.tracker,
        )
    return model.predict(**common_options)


def draw_fps(frame, fps):
    """Draw a small FPS counter in the top-left corner."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 255, 50),
        1,
        cv2.LINE_AA,
    )


def save_screenshot(frame):
    """Save the current annotated frame with a timestamped filename."""
    SCREENSHOT_FOLDER.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    screenshot_file = SCREENSHOT_FOLDER / f"aquarium_live_{timestamp}.jpg"

    if cv2.imwrite(str(screenshot_file), frame):
        print(f"Screenshot saved: {screenshot_file}")
    else:
        print(f"WARNING: Could not save screenshot: {screenshot_file}")


def stop_process(process):
    """Stop FFmpeg cleanly, escalating to kill only if it does not exit."""
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def resize_to_width(frame, target_width):
    """Resize a local-video frame while preserving its aspect ratio."""
    source_height, source_width = frame.shape[:2]
    if source_width == target_width:
        return frame

    target_height = round((source_height * target_width / source_width) / 2) * 2
    interpolation = cv2.INTER_AREA if target_width < source_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def open_local_video(video_file):
    """Open a local video and return its capture and playback FPS."""
    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open local video: {video_file}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
        print("WARNING: Video FPS was unavailable; using 30 FPS for playback.")
    return capture, source_fps


def open_usb_camera(args):
    """Open a Windows USB camera with DirectShow, then apply requested settings."""
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    backend_name = "DirectShow"

    if not capture.isOpened():
        capture.release()
        print(
            f"WARNING: DirectShow could not open camera {args.camera}; "
            "trying OpenCV's default backend."
        )
        capture = cv2.VideoCapture(args.camera)
        backend_name = "OpenCV default"

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open USB camera ID {args.camera}")

    if args.camera_width is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height is not None:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps is not None:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"USB camera {args.camera} opened with: {backend_name}")
    print(
        f"Camera capture: {actual_width}x{actual_height} | "
        f"reported FPS: {actual_fps:.3f}"
    )
    return capture


def read_local_video_frame(capture, loop_video):
    """Read one local frame, optionally restarting once the video ends."""
    success, frame = capture.read()
    if success:
        return frame, False

    if not loop_video:
        return None, False

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, frame = capture.read()
    if not success:
        raise RuntimeError("Local video could not be read after restarting")
    return frame, True


def run_live_view(args):
    """Run the shared live display loop for YouTube or a local video."""
    ffmpeg_process = None
    video_capture = None
    frame_width = args.width
    frame_height = None
    frame_byte_count = None
    source_fps = None

    try:
        if args.source_type == "youtube":
            m3u8_url = get_m3u8_url(args.url)
            frame_width, frame_height = get_stream_size(m3u8_url, args.width)
            frame_byte_count = frame_width * frame_height * 3
            ffmpeg_process = start_ffmpeg(m3u8_url, frame_width)
            if ffmpeg_process.stdout is None:
                raise RuntimeError("FFmpeg stdout pipe was not created")
            print(f"YouTube/FFmpeg frame size: {frame_width}x{frame_height}")
        elif args.source_type == "video":
            video_capture, source_fps = open_local_video(args.video)
            source_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if source_width <= 0 or source_height <= 0:
                raise RuntimeError("Local video has invalid frame dimensions")
            frame_height = round((source_height * frame_width / source_width) / 2) * 2
            print(f"Local video: {args.video}")
            print(
                f"Display frame size: {frame_width}x{frame_height} | "
                f"playback FPS: {source_fps:.3f}"
            )
        else:
            video_capture = open_usb_camera(args)
            print(f"Camera display width after aspect-ratio resize: {frame_width}")

        print(f"Loading model: {args.model}")
        model = YOLO(str(args.model))
        label_font, use_chinese = load_label_font(frame_width)

        if args.roi is not None:
            print(f"Detection ROI (normalized): {args.roi}")
            print(
                "Only the ROI is sent to YOLO. Areas outside it are ignored "
                "to reduce false positives from algae, mesh, black background, "
                "and bottom reflections."
            )
        else:
            print("Detection ROI: full frame")

        if args.track:
            print(f"Inference mode: tracking with {args.tracker}")
        else:
            print("Inference mode: predict")
        print(f"YOLO processing interval: every {args.process_every} frame(s)")
        print("Controls: q = quit, s = save screenshot")

        previous_time = time.perf_counter()
        smoothed_fps = 0.0
        frame_number = 0
        last_result = None
        next_video_deadline = time.perf_counter()

        while True:
            restarted_video = False
            if args.source_type == "youtube":
                raw_frame = read_exactly(ffmpeg_process.stdout, frame_byte_count)
                if raw_frame is None:
                    return_code = ffmpeg_process.poll()
                    raise RuntimeError(
                        "FFmpeg stream ended or stopped producing frames "
                        f"(exit code: {return_code})"
                    )
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    (frame_height, frame_width, 3)
                )
                # frombuffer creates a read-only view; copy it before drawing.
                frame = frame.copy()
            elif args.source_type == "video":
                frame, restarted_video = read_local_video_frame(
                    video_capture,
                    args.loop_video,
                )
                if frame is None:
                    print("Local video reached the end.")
                    break
                frame = resize_to_width(frame, frame_width)
                if restarted_video:
                    frame_number = 0
                    last_result = None
                    next_video_deadline = time.perf_counter()
            else:
                success, frame = video_capture.read()
                if not success or frame is None:
                    raise RuntimeError(
                        f"USB camera {args.camera} stopped producing frames"
                    )
                frame = resize_to_width(frame, frame_width)

            roi_x1, roi_y1, roi_x2, roi_y2 = roi_pixel_bounds(frame, args.roi)

            should_process = last_result is None or frame_number % args.process_every == 0
            if should_process:
                inference_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                # YOLO sees only this crop. Drawing offsets its boxes back to
                # full-frame coordinates.
                last_result = run_inference(model, inference_frame, args)[0]

            frame = draw_detections(
                frame,
                last_result,
                roi_x1,
                roi_y1,
                label_font,
                use_chinese,
                args.hide_conf,
                args.hide_id,
                args.box_thickness,
            )

            if args.draw_roi and args.roi is not None:
                cv2.rectangle(
                    frame,
                    (roi_x1, roi_y1),
                    (roi_x2 - 1, roi_y2 - 1),
                    (0, 255, 255),
                    1,
                )

            if args.stats:
                frame = draw_species_stats(
                    frame,
                    last_result,
                    label_font,
                    use_chinese,
                )

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else 0.90 * smoothed_fps + 0.10 * instant_fps
            )
            draw_fps(frame, smoothed_fps)

            # Display resizing happens only after inference and every overlay
            # are complete. It does not change the frame sent to YOLO.
            display_frame = (
                resize_to_width(frame, args.display_width)
                if args.display_width is not None
                else frame
            )

            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_screenshot(display_frame)

            frame_number += 1

            # Local files are decoded faster than real time. Hold each frame
            # until its source-FPS deadline. If inference is slower, do not add
            # extra delay.
            if args.source_type == "video":
                next_video_deadline += 1.0 / source_fps
                delay = next_video_deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -(1.0 / source_fps):
                    next_video_deadline = time.perf_counter()

    finally:
        if ffmpeg_process is not None and ffmpeg_process.stdout is not None:
            ffmpeg_process.stdout.close()
        stop_process(ffmpeg_process)
        if video_capture is not None:
            video_capture.release()
        cv2.destroyAllWindows()


def main():
    args = parse_arguments()
    try:
        check_settings(args)
        run_live_view(args)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
