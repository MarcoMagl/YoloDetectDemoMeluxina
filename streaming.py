#!/usr/bin/env python3
"""
Copyright (c) 2026 LuxProvide S.A.

This code is the property of LuxProvide S.A. and was developed by
Marco Magliulo.

Permission is granted to reuse, modify, and redistribute this code,
with or without changes, provided that this notice and attribution
are retained.
"""
import subprocess
import cv2
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO
import subprocess
import shlex


# ============================================================================
# Input: YouTube live stream URL
# ============================================================================
# This URL is used as the video source. You can change it if you want 
#The stream is resolved to a direct media URL using yt-dlp.
url = "https://www.youtube.com/watch?v=u7GyFcQJs98"


# ============================================================================
# Resolve the YouTube stream to a direct video URL
# ============================================================================
result = subprocess.run(
    ["python", "-m", "yt_dlp", "-f", "best[ext=mp4]/best", "-g", url],
    capture_output=True,
    text=True,
    check=True
)

# The direct streaming URL returned by yt-dlp
stream_url = result.stdout.strip()
print("Streaming from:", stream_url)


# ============================================================================
# OpenCV video capture
# ============================================================================
# Open the resolved stream using the FFmpeg backend.
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)


# ============================================================================
# YOLO model initialization
# ============================================================================
# Load a pretrained YOLO model.
# you can change this model if you want 
model = YOLO("yolo11s.pt")

# Dictionary mapping class IDs to class names
class_names = model.names

# Subset of COCO classes relevant to a CCTV / traffic-like scenario
# If you change the stream make sure you capture relevant calsses 
wanted_names = [
    "person", "bicycle", "car", "motorcycle",
    "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant",
    "stop sign", "parking meter",
    "bench", "bird", "cat", "dog"
]

# Convert class names to numeric class IDs expected by the model
allowed_classes = [
    cls_id for cls_id, name in class_names.items()
    if name in wanted_names
]

print("Allowed classes:", allowed_classes)


# ============================================================================
# Flask application for MJPEG streaming
# ============================================================================
# A lightweight HTTP server is used to stream annotated frames by YOLO as MJPEG.
# I had to do that because I could not get a X11 window 
# Using flask allows us to avoid this step and we can reach the HTTP server with
# port forwarding easily ! 
app = Flask(__name__)


def generate_frames():
    """
    Generator function producing JPEG-encoded frames for MJPEG streaming.
    Each frame is:
      - read from the video stream
      - processed by YOLO
      - annotated with bounding boxes
      - encoded as JPEG
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference on the current frame
        # look at the doc of model to understand the effect of the different 
        # parameter we have here 
        results = model(
            frame,
            conf=0.1,
            classes=allowed_classes,
            device=0,
            imgsz=960,
            verbose=False
        )

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Encode the frame as JPEG (required for MJPEG streaming)
        success, buffer = cv2.imencode(
            ".jpg",
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # Yield a single MJPEG frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes +
            b"\r\n"
        )


@app.route("/video")
def video():
    """
    HTTP endpoint exposing the MJPEG stream on the port we choose.

    We can then view this stream using SSH port forwarding !
    """
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# Utility: print SSH port-forwarding command
# the user just has to copy and paste the command we get 
# ============================================================================
def print_ssh_forwarding_command(port=5000):
    """
    Prints the SSH command that the user must run on their *local machine*
    to forward the remote MJPEG stream to localhost.
    """

    # Obtain the hostname of the compute node running this script
    hostname = subprocess.check_output(
        ["hostname"],
        text=True
    ).strip()

    # Obtain the current username
    user = subprocess.check_output(
        ["whoami"],
        text=True
    ).strip()

    # Construct the SSH command for port forwarding
    ssh_command = (
        f"ssh -N -v "
        f"-p 8822 "
        f"-i ~/.ssh/id_ed25519_mlux "
        f"-L {port}:{hostname}:{port} "
        f"{user}@login.lxp.lu"
    )

    # Print user-facing instructions
    print("\n" + "=" * 80)
    print("To stream the live YOLO output on your LOCAL machine, run:")
    print()
    print(ssh_command)
    print()
    print("Then open in your browser:")
    print(f"  http://localhost:{port}/video")
    print("=" * 80 + "\n")


# ============================================================================
# Main entry point
# ============================================================================
if __name__ == "__main__":

    # Display the SSH port-forwarding command at startup
    print_ssh_forwarding_command(port=5000)

    # Start the Flask server (listening on all interfaces)
    app.run(host="0.0.0.0", port=5000, threaded=True)
