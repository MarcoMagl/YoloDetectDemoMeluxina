import subprocess
import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import subprocess
import shlex

# ----------------------
# YouTube live stream URL
# ----------------------
url = "https://www.youtube.com/watch?v=u7GyFcQJs98"

# Use yt-dlp via Python to avoid PATH issues (container-safe)
result = subprocess.run(
    ["python", "-m", "yt_dlp", "-f", "best[ext=mp4]/best", "-g", url],
    capture_output=True,
    text=True,
    check=True
)

stream_url = result.stdout.strip()
print("Streaming from:", stream_url)

cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

# ----------------------
# YOLO model
# ----------------------
model = YOLO("yolo11s.pt")

class_names = model.names

wanted_names = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant",
    "stop sign", "parking meter",
    "bench", "bird", "cat", "dog"
]

allowed_classes = [
    cls_id for cls_id, name in class_names.items()
    if name in wanted_names
]

print("Allowed classes:", allowed_classes)

# ----------------------
# Flask MJPEG server
# ----------------------
app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(
            frame,
            conf=0.1,
            classes=allowed_classes,
            device=0,
            imgsz=960,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # Encode as JPEG
        success, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG chunk
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def print_ssh_forwarding_command(port=5000):
    # Get current hostname
    hostname = subprocess.check_output(
        ["hostname"],
        text=True
    ).strip()

    # Get current user
    user = subprocess.check_output(
        ["whoami"],
        text=True
    ).strip()

    ssh_command = (
        f"ssh -N -v "
        f"-p 8822 "
        f"-i ~/.ssh/id_ed25519_mlux "
        f"-L {port}:{hostname}:{port} "
        f"{user}@login.lxp.lu"
    )

    print("\n" + "=" * 80)
    print("To stream the live YOLO output on your LOCAL machine, run:")
    print()
    print(ssh_command)
    print()
    print(f"Then open in your browser:")
    print(f"  http://localhost:{port}/video")
    print("=" * 80 + "\n")


if __name__ == "__main__":

    print_ssh_forwarding_command(port=5000)

    app.run(host="0.0.0.0", port=5000, threaded=True)




