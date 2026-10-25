from pathlib import Path
from collections import deque
from datetime import datetime
import argparse
import logging
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default paths and settings
# ---------------------------------------------------------------------------
DEFAULT_URL = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "reef_ai_yolov12n_v2_full_tank.pt"
DEFAULT_WORK_DIR = PROJECT_ROOT / "outputs" / "delayed_live_as_ready"
RENDER_SCRIPT = Path(__file__).resolve().parent / "render_demo_video.py"
FFMPEG_EXE = "ffmpeg"
WINDOW_NAME = "Delayed AI Live"


def parse_arguments():
    """Read command-line options."""
    parser = argparse.ArgumentParser(
        description="Capture, annotate, and play short YouTube Live chunks as ready."
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="YouTube Live URL")
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=10.0,
        help="Length of each raw chunk in seconds (default: 10)",
    )
    parser.add_argument(
        "--capture_width",
        type=int,
        default=800,
        help="Width encoded into raw chunks (default: 800)",
    )
    parser.add_argument(
        "--capture_fps",
        type=float,
        default=10.0,
        help="Frames per second encoded into raw chunks (default: 10)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Playback display width (default: 800)",
    )
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--track", action="store_true")
    parser.add_argument(
        "--no_track",
        action="store_true",
        help="Do not pass --track to the renderer, even if --track is present",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--hide_id", action="store_true")
    parser.add_argument("--hide_conf", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--box_thickness", type=int, default=2)
    parser.add_argument("--output_fps", type=float, default=30.0)
    parser.add_argument("--max_pending_raw", type=int, default=3)
    parser.add_argument("--max_pending_processed", type=int, default=3)
    parser.add_argument(
        "--keep_raw",
        action="store_true",
        help="Keep successfully captured raw chunks after processing",
    )
    return parser.parse_args()


def validate_arguments(args):
    """Validate local paths and numeric settings before starting threads."""
    if not args.url:
        raise ValueError("--url is required for YouTube Live capture")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model was not found: {args.model}")
    if not RENDER_SCRIPT.is_file():
        raise FileNotFoundError(f"Renderer was not found: {RENDER_SCRIPT}")
    if not shutil.which(FFMPEG_EXE):
        raise FileNotFoundError("FFmpeg was not found on PATH")
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk_seconds must be greater than 0")
    if args.capture_width <= 0:
        raise ValueError("--capture_width must be greater than 0")
    if args.capture_fps <= 0:
        raise ValueError("--capture_fps must be greater than 0")
    if args.width <= 0:
        raise ValueError("--width must be greater than 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if args.box_thickness <= 0:
        raise ValueError("--box_thickness must be greater than 0")
    if args.output_fps <= 0:
        raise ValueError("--output_fps must be greater than 0")
    if args.max_pending_raw <= 0:
        raise ValueError("--max_pending_raw must be greater than 0")
    if args.max_pending_processed <= 0:
        raise ValueError("--max_pending_processed must be greater than 0")
    if not args.tracker.strip():
        raise ValueError("--tracker cannot be empty")


def create_work_folders(work_dir):
    """Create the requested pipeline folder structure."""
    folders = {
        "raw": work_dir / "raw",
        "processed": work_dir / "processed",
        "screenshots": work_dir / "screenshots",
        "logs": work_dir / "logs",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def configure_logging(log_folder):
    """Log to both the console and a timestamped UTF-8 file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_folder / f"delayed_live_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("Log file: %s", log_file)


class ChildProcessRegistry:
    """Keep track of active child processes so shutdown can terminate them."""

    def __init__(self):
        self._processes = set()
        self._lock = threading.Lock()

    def add(self, process):
        with self._lock:
            self._processes.add(process)

    def remove(self, process):
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self):
        with self._lock:
            processes = list(self._processes)

        for process in processes:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass

        deadline = time.monotonic() + 3.0
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass


class PipelineQueues:
    """Thread-safe bounded queues with oldest-item eviction."""

    def __init__(self, max_raw, max_processed):
        self.raw = deque()
        self.processed = deque()
        self.max_raw = max_raw
        self.max_processed = max_processed
        self.condition = threading.Condition()

    def add_raw(self, item):
        dropped = None
        with self.condition:
            if len(self.raw) >= self.max_raw:
                dropped = self.raw.popleft()
            self.raw.append(item)
            self.condition.notify_all()
        return dropped

    def get_raw(self, stop_event):
        with self.condition:
            while not self.raw and not stop_event.is_set():
                self.condition.wait(timeout=0.5)
            if self.raw:
                return self.raw.popleft()
        return None

    def add_processed(self, item):
        dropped = None
        with self.condition:
            if len(self.processed) >= self.max_processed:
                dropped = self.processed.popleft()
            self.processed.append(item)
            self.condition.notify_all()
        return dropped

    def get_processed_nowait(self):
        with self.condition:
            if self.processed:
                return self.processed.popleft()
        return None

    def sizes(self):
        with self.condition:
            return len(self.raw), len(self.processed)

    def wake_all(self):
        with self.condition:
            self.condition.notify_all()


def safe_unlink(file_path):
    """Remove one generated file if it exists, without raising on failure."""
    try:
        file_path.unlink(missing_ok=True)
    except OSError as error:
        logging.warning("Could not remove %s: %s", file_path, error)


def run_process(command, registry, stop_event, capture_output=False):
    """Run one child process while making it available for clean shutdown."""
    output_setting = subprocess.PIPE if capture_output else None
    process = subprocess.Popen(
        command,
        stdout=output_setting,
        stderr=output_setting,
        text=capture_output,
        encoding="utf-8" if capture_output else None,
        errors="replace" if capture_output else None,
    )
    registry.add(process)

    try:
        while process.poll() is None:
            if stop_event.wait(0.2):
                try:
                    process.terminate()
                except OSError:
                    pass
                break

        if capture_output:
            stdout_text, stderr_text = process.communicate()
        else:
            process.wait()
            stdout_text, stderr_text = "", ""
        return process.returncode, stdout_text, stderr_text
    finally:
        registry.remove(process)


def resolve_stream_url(youtube_url, registry, stop_event):
    """Use yt-dlp to resolve the current direct YouTube Live stream URL."""
    command = [
        "yt-dlp",
        "-f",
        "best[protocol*=m3u8]/best",
        "-g",
        youtube_url,
    ]
    return_code, stdout_text, stderr_text = run_process(
        command,
        registry,
        stop_event,
        capture_output=True,
    )
    if stop_event.is_set():
        return None
    if return_code != 0:
        raise RuntimeError(f"yt-dlp failed: {stderr_text.strip()}")

    direct_urls = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    if not direct_urls:
        raise RuntimeError("yt-dlp did not return a direct stream URL")
    return direct_urls[0]


def capture_one_chunk(
    stream_url,
    temporary_file,
    chunk_seconds,
    capture_width,
    capture_fps,
    registry,
    stop_event,
):
    """Record one short MP4 chunk from the direct stream URL."""
    command = [
        str(FFMPEG_EXE),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        stream_url,
        "-t",
        str(chunk_seconds),
        "-vf",
        f"fps={capture_fps:g},scale={capture_width}:-2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_file),
    ]
    return_code, _stdout_text, _stderr_text = run_process(
        command,
        registry,
        stop_event,
    )
    return return_code == 0 and temporary_file.is_file() and temporary_file.stat().st_size > 0


def capture_worker(args, folders, queues, registry, stop_event):
    """Continuously resolve/capture chunks and add completed files to raw queue."""
    chunk_number = 1
    stream_url = None

    while not stop_event.is_set():
        if stream_url is None:
            try:
                logging.info("Resolving YouTube Live stream URL...")
                stream_url = resolve_stream_url(args.url, registry, stop_event)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                logging.warning("Could not resolve stream URL: %s", error)
                stop_event.wait(3.0)
                continue

        final_file = folders["raw"] / f"raw_{chunk_number:06d}.mp4"
        temporary_file = folders["raw"] / f"raw_{chunk_number:06d}_capturing.mp4"
        safe_unlink(temporary_file)

        try:
            success = capture_one_chunk(
                stream_url,
                temporary_file,
                args.chunk_seconds,
                args.capture_width,
                args.capture_fps,
                registry,
                stop_event,
            )
        except (OSError, subprocess.SubprocessError) as error:
            logging.warning("FFmpeg exception for chunk #%d: %s", chunk_number, error)
            success = False

        if stop_event.is_set():
            safe_unlink(temporary_file)
            break

        if not success:
            logging.warning("FFmpeg failed for chunk #%d; retrying soon", chunk_number)
            safe_unlink(temporary_file)
            # A signed stream URL may have expired. Resolve a fresh one.
            stream_url = None
            stop_event.wait(3.0)
            continue

        try:
            temporary_file.replace(final_file)
        except OSError as error:
            logging.warning("Could not finalize chunk #%d: %s", chunk_number, error)
            safe_unlink(temporary_file)
            stop_event.wait(1.0)
            continue

        logging.info("Captured chunk #%d", chunk_number)
        dropped = queues.add_raw((chunk_number, final_file))
        if dropped is not None:
            dropped_number, dropped_file = dropped
            safe_unlink(dropped_file)
            logging.info("Dropped old raw chunk #%d", dropped_number)

        chunk_number += 1


def renderer_supports_tracker():
    """Detect whether the existing renderer currently accepts --tracker."""
    try:
        renderer_text = RENDER_SCRIPT.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    return '"--tracker"' in renderer_text or "'--tracker'" in renderer_text


def build_render_command(args, raw_file, processed_file, tracker_supported):
    """Build a Windows-safe argument list for render_demo_video.py."""
    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--source",
        str(raw_file),
        "--output",
        str(processed_file),
        "--model",
        str(args.model),
        "--box_thickness",
        str(args.box_thickness),
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--output_fps",
        str(args.output_fps),
    ]
    if args.track and not args.no_track:
        command.append("--track")
        if tracker_supported:
            command.extend(["--tracker", args.tracker])
    if args.hide_id:
        command.append("--hide_id")
    if args.hide_conf:
        command.append("--hide_conf")
    if args.stats:
        command.append("--stats")
    return command


def processing_worker(args, folders, queues, registry, stop_event):
    """Process raw chunks in queue order and enqueue completed AI clips."""
    tracker_supported = renderer_supports_tracker()
    tracking_enabled = args.track and not args.no_track
    if tracking_enabled and not tracker_supported and args.tracker != "bytetrack.yaml":
        logging.warning(
            "render_demo_video.py does not currently accept --tracker; "
            "it will use its built-in ByteTrack setting instead of %s",
            args.tracker,
        )

    while not stop_event.is_set():
        raw_item = queues.get_raw(stop_event)
        if raw_item is None:
            continue

        chunk_number, raw_file = raw_item
        processed_file = folders["processed"] / f"ai_{chunk_number:06d}.mp4"
        safe_unlink(processed_file)
        logging.info("Processing chunk #%d", chunk_number)

        command = build_render_command(
            args,
            raw_file,
            processed_file,
            tracker_supported,
        )
        try:
            return_code, _stdout_text, _stderr_text = run_process(
                command,
                registry,
                stop_event,
            )
        except (OSError, subprocess.SubprocessError) as error:
            logging.warning("Processing exception for chunk #%d: %s", chunk_number, error)
            return_code = -1

        if not args.keep_raw:
            safe_unlink(raw_file)

        if stop_event.is_set():
            safe_unlink(processed_file)
            break

        if return_code != 0 or not processed_file.is_file():
            logging.warning("Processing failed for chunk #%d; continuing", chunk_number)
            safe_unlink(processed_file)
            continue

        logging.info("Processed chunk #%d", chunk_number)
        dropped = queues.add_processed((chunk_number, processed_file))
        if dropped is not None:
            dropped_number, dropped_file = dropped
            safe_unlink(dropped_file)
            logging.info("Dropped old processed chunk #%d", dropped_number)


def resize_to_width(frame, target_width):
    """Resize one playback frame while preserving aspect ratio."""
    source_height, source_width = frame.shape[:2]
    if source_width == target_width:
        return frame
    target_height = max(1, round(source_height * target_width / source_width))
    interpolation = cv2.INTER_AREA if target_width < source_width else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def save_screenshot(frame, screenshot_folder):
    """Save exactly the frame currently shown in the playback window."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    screenshot_file = screenshot_folder / f"delayed_ai_{timestamp}.jpg"
    if cv2.imwrite(str(screenshot_file), frame):
        logging.info("Screenshot saved: %s", screenshot_file)
    else:
        logging.warning("Could not save screenshot: %s", screenshot_file)


def make_waiting_frame(display_width, raw_count, processed_count):
    """Create the waiting screen shown while no processed clip is ready."""
    display_height = max(360, round(display_width * 9 / 16))
    frame = np.full((display_height, display_width, 3), (20, 24, 28), dtype=np.uint8)
    lines = [
        "Preparing delayed AI stream...",
        "Waiting for next AI clip...",
        f"Raw queue: {raw_count}",
        f"Processed queue: {processed_count}",
    ]
    y = max(70, display_height // 2 - 65)
    for index, line in enumerate(lines):
        scale = 0.75 if index < 2 else 0.60
        color = (230, 230, 230) if index < 2 else (120, 220, 255)
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        x = max(15, (display_width - text_size[0]) // 2)
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 38
    return frame


def handle_display_key(key, displayed_frame, screenshot_folder, stop_event):
    """Handle q and s for both waiting and clip playback screens."""
    if key == ord("q"):
        stop_event.set()
        return False
    if key == ord("s"):
        save_screenshot(displayed_frame, screenshot_folder)
    return True


def play_processed_clip(
    chunk_number,
    clip_file,
    args,
    screenshot_folder,
    stop_event,
):
    """Play one processed clip smoothly and return after it finishes."""
    capture = cv2.VideoCapture(str(clip_file))
    if not capture.isOpened():
        logging.warning("Could not open processed chunk #%d: %s", chunk_number, clip_file)
        return

    logging.info("Playing chunk #%d", chunk_number)
    frame_interval = 1.0 / args.output_fps
    next_deadline = time.perf_counter()

    try:
        while not stop_event.is_set():
            success, frame = capture.read()
            if not success:
                break

            displayed_frame = resize_to_width(frame, args.width)
            cv2.imshow(WINDOW_NAME, displayed_frame)
            key = cv2.waitKey(1) & 0xFF
            if not handle_display_key(
                key,
                displayed_frame,
                screenshot_folder,
                stop_event,
            ):
                break

            next_deadline += frame_interval
            delay = next_deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -frame_interval:
                next_deadline = time.perf_counter()
    finally:
        capture.release()


def playback_loop(args, folders, queues, stop_event):
    """Run waiting-screen and as-soon-as-ready playback on the main thread."""
    last_displayed_frame = None
    waiting_refresh_deadline = 0.0

    while not stop_event.is_set():
        processed_item = queues.get_processed_nowait()
        if processed_item is not None:
            chunk_number, clip_file = processed_item
            play_processed_clip(
                chunk_number,
                clip_file,
                args,
                folders["screenshots"],
                stop_event,
            )
            safe_unlink(clip_file)
            continue

        current_time = time.monotonic()
        if current_time >= waiting_refresh_deadline or last_displayed_frame is None:
            raw_count, processed_count = queues.sizes()
            last_displayed_frame = make_waiting_frame(
                args.width,
                raw_count,
                processed_count,
            )
            waiting_refresh_deadline = current_time + 0.25

        cv2.imshow(WINDOW_NAME, last_displayed_frame)
        key = cv2.waitKey(30) & 0xFF
        handle_display_key(
            key,
            last_displayed_frame,
            folders["screenshots"],
            stop_event,
        )


def main():
    args = parse_arguments()
    stop_event = threading.Event()
    registry = ChildProcessRegistry()
    threads = []

    try:
        validate_arguments(args)
        folders = create_work_folders(args.work_dir)
        configure_logging(folders["logs"])
        logging.info("Capture width: %s", args.capture_width)
        logging.info("Capture fps: %s", f"{args.capture_fps:g}")
        logging.info("Chunk seconds: %s", f"{args.chunk_seconds:g}")
        if args.track and args.no_track:
            logging.info("--no_track overrides --track; rendering without tracking")
        queues = PipelineQueues(
            args.max_pending_raw,
            args.max_pending_processed,
        )

        capture_thread = threading.Thread(
            target=capture_worker,
            args=(args, folders, queues, registry, stop_event),
            name="capture-thread",
            daemon=True,
        )
        processing_thread = threading.Thread(
            target=processing_worker,
            args=(args, folders, queues, registry, stop_event),
            name="processing-thread",
            daemon=True,
        )
        threads = [capture_thread, processing_thread]
        for thread in threads:
            thread.start()

        playback_loop(args, folders, queues, stop_event)

    except KeyboardInterrupt:
        print("\nStopping delayed live pipeline...")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    finally:
        stop_event.set()
        if "queues" in locals():
            queues.wake_all()
        registry.terminate_all()
        for thread in threads:
            thread.join(timeout=5.0)
        cv2.destroyAllWindows()
        if logging.getLogger().handlers:
            logging.info("Delayed live pipeline stopped")


if __name__ == "__main__":
    main()
