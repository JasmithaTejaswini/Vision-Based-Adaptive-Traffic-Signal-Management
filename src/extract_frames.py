
import cv2
import json
import csv
import argparse
from pathlib import Path

import numpy as np


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "road_regions.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"


def open_video(video_path):
    """Open a video and return its capture object and metadata."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError("Video metadata is invalid.")

    metadata = {
        "video_path": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps,
    }

    return cap, metadata


def preprocess_frame(frame):
    """Prepare a frame for downstream vehicle detection."""
    # Keep original resolution; convert to a consistent color format.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Light denoising without heavily changing vehicle appearance.
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame


def load_region_config(config_path, width, height):
    """Load ROIs and validate coordinates against video dimensions."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Region configuration not found: {config_path}. "
            "Run this program with --setup-rois first."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("frame_width") != width or config.get("frame_height") != height:
        raise ValueError(
            "ROI configuration resolution does not match the video. "
            "Reconfigure ROIs for this camera view."
        )

    for road_name, region in config["roads"].items():
        for region_type in ("road_roi", "waiting_zone"):
            points = region[region_type]

            if len(points) < 3:
                raise ValueError(
                    f"{road_name}/{region_type} needs at least 3 points."
                )

            for x, y in points:
                if not (0 <= x < width and 0 <= y < height):
                    raise ValueError(
                        f"Invalid coordinate {(x, y)} in "
                        f"{road_name}/{region_type}."
                    )

    return config


def select_polygon(frame, title):
    """Let the user click polygon vertices on the actual video frame."""
    points = []
    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            display = frame.copy()

            for point in points:
                cv2.circle(display, point, 5, (0, 255, 0), -1)

            if len(points) > 1:
                cv2.polylines(
                    display,
                    [np.array(points, dtype=np.int32)],
                    False,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(title, display)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_callback)

    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(20) & 0xFF

        # Enter accepts the polygon; R resets it; Esc cancels.
        if key == 13 and len(points) >= 3:
            break
        elif key == ord("r"):
            points.clear()
            display = frame.copy()
        elif key == 27:
            points.clear()
            break

    cv2.destroyWindow(title)
    return [[int(x), int(y)] for x, y in points]


def setup_rois(video_path, config_path):
    """Interactively configure road ROIs and waiting zones."""
    cap, metadata = open_video(video_path)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read the first frame.")

    roads = {}

    print("\nDraw each polygon by clicking its corner points.")
    print("Press ENTER to accept, R to reset, ESC to cancel.")
    print("Use the actual visible road boundaries and stop-line areas.\n")

    for road in ("north", "east", "south", "west"):
        road_roi = select_polygon(frame, f"{road.upper()}: ROAD ROI")

        if len(road_roi) < 3:
            print(f"Cancelled while configuring {road}.")
            return

        waiting_zone = select_polygon(
            frame, f"{road.upper()}: WAITING ZONE"
        )

        if len(waiting_zone) < 3:
            print(f"Cancelled while configuring {road}.")
            return

        roads[road] = {
            "road_roi": road_roi,
            "waiting_zone": waiting_zone,
        }

    config = {
        "frame_width": metadata["width"],
        "frame_height": metadata["height"],
        "roads": roads,
        "note": (
            "Coordinates are pixel locations. Verify each polygon against "
            "the camera view. Do not use an obscured road as if fully visible."
        ),
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"\nROI configuration saved to: {config_path}")


def process_video(video_path, config_path, sample_interval=5):
    """Extract timestamped frames and create ROI visualization."""
    cap, metadata = open_video(video_path)
    config = load_region_config(
        config_path, metadata["width"], metadata["height"]
    )

    frames_dir = OUTPUT_ROOT / "frames"
    preview_dir = OUTPUT_ROOT / "roi_previews"

    frames_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    timestamp_csv = OUTPUT_ROOT / "frame_timestamps.csv"
    interval_frames = max(1, round(metadata["fps"] * sample_interval))

    with open(timestamp_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "timestamp_seconds", "filename"])

        frame_index = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_index % interval_frames == 0:
                timestamp = frame_index / metadata["fps"]
                filename = f"frame_{frame_index:06d}.jpg"
                output_path = frames_dir / filename

                processed = preprocess_frame(frame)

                # Save RGB-preprocessed frame as an image.
                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                )

                # Visualize road ROIs and waiting zones.
                preview = frame.copy()

                for road_name, region in config["roads"].items():
                    road_polygon = np.array(
                        region["road_roi"], dtype=np.int32
                    )
                    waiting_polygon = np.array(
                        region["waiting_zone"], dtype=np.int32
                    )

                    cv2.polylines(
                        preview, [road_polygon], True, (0, 255, 0), 2
                    )
                    cv2.polylines(
                        preview, [waiting_polygon], True, (0, 0, 255), 2
                    )

                    x, y = region["road_roi"][0]
                    cv2.putText(
                        preview,
                        road_name.upper(),
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imwrite(
                    str(preview_dir / filename),
                    preview,
                )

                writer.writerow([
                    frame_index,
                    round(timestamp, 3),
                    filename,
                ])

                saved_count += 1

            frame_index += 1

    cap.release()

    metadata["sample_interval_seconds"] = sample_interval
    metadata["sampled_frames_saved"] = saved_count

    with open(OUTPUT_ROOT / "video_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nVideo processing complete.")
    print(f"Frames saved: {saved_count}")
    print(f"Timestamp index: {timestamp_csv}")
    print(f"ROI previews: {preview_dir}")
    print(f"Metadata: {OUTPUT_ROOT / 'video_metadata.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Traffic video preparation and ROI configuration."
    )
    parser.add_argument(
        "--video",
        default="data/raw/traffic_video_1.mp4",
        help="Path to the input video, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--setup-rois",
        action="store_true",
        help="Interactively configure road ROIs and waiting zones.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5,
        help="Seconds between saved frames.",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if args.interval <= 0:
        raise ValueError("Frame extraction interval must be positive.")

    if args.setup_rois:
        setup_rois(video_path, CONFIG_PATH)
    else:
        process_video(video_path, CONFIG_PATH, args.interval)


if __name__ == "__main__":
    main()