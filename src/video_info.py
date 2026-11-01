
import cv2

VIDEO_PATH = "data/raw/traffic_video_1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps if fps > 0 else 0

print("Video opened successfully!")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration / 60:.2f} minutes")

ret, frame = cap.read()

if ret:
    cv2.imshow("First Video Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not read the first frame.")

cap.release()
