
import cv2
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "data/raw/traffic_video_1.mp4"
CONFIG_PATH = ROOT / "configs/road_regions.json"
OUTPUT_PATH = ROOT / "data/processed/roi_previews/final_roi_check.jpg"

# Load existing configuration
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Read an actual video frame
cap = cv2.VideoCapture(str(VIDEO_PATH))
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read video frame.")

if frame.shape[1] != config["frame_width"] or frame.shape[0] != config["frame_height"]:
    raise ValueError("Video resolution does not match ROI configuration.")

# Draw ROIs
overlay = frame.copy()

for direction, zones in config["roads"].items():
    road = zones.get("road_roi")
    waiting = zones.get("waiting_zone")

    if road:
        pts = np.array(road, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))

    if waiting:
        pts = np.array(waiting, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 220))

# Blend transparency
frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)

# Draw outlines and labels
for direction, zones in config["roads"].items():
    road = zones.get("road_roi")
    waiting = zones.get("waiting_zone")

    if road:
        pts = np.array(road, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        x, y = pts[0]
        cv2.putText(frame, direction.upper(), (x, max(y - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if waiting:
        pts = np.array(waiting, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), frame)

print("Preview saved:", OUTPUT_PATH)
