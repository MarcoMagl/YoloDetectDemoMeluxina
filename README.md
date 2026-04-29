# Yolo Detect demo on Meluxina

![YOLO running on a YouTube live stream](demo.gif)

## Content

This repository demonstrates how to run a **real-time object detection inference** using the **YOLO framework by Ultralytics** on the **MeluXina supercomputer**, and how to **stream the live annotated output to a local machine** using SSH port forwarding (no X11 required).

The example performs object detection on a **YouTube live stream** using Meluxina  **GPU nodes**, and exposes the output through a HTTP video stream.

Of course, you can run this example on another GPU equipped machine but the command to access the platform and launch jobs might be different. 

***

## Step 1 — Connect to MeluXina and allocate a GPU job

For more info, see the dedicateed [LuxProvide documentation](https://docs.lxp.lu/first-steps/connecting/)

First, connect to MeluXina:

```bash
ssh '<your-user-ID>'@login.lxp.lu -p 8822 -i ~/.ssh/id_ed25519_mlux
```

Then allocate an interactive job on the GPU partition, for example:

```bash
srun -A <yourAccount> -p gpu -q default -N 1 -n 1 --time=01:00:00 --pty bash -i
```

Once the allocation is granted, a shell will be spawned on a GPU compute node.

***

## Step 2 — Set up the software environment

Simply run:

```bash
bash setup_env.sh
```

This script will:

*   Load the required **MeluXina software modules** (based on `LMod`)
*   Create and activate a **dedicated Python environment**
*   Install the necessary Python packages (YOLO, OpenCV, Flask, etc.)

***

## Step 3 — Run the inference

Once the environment is ready, start the inference with:

```bash
python run_inference.py
```

When the script starts, it will:

*   Resolve a YouTube live stream URL using `yt-dlp` --> **Feel free to change the stream URL to the one of your liking**
*   Open the stream using FFmpeg
*   Run YOLO inference on each frame using the NVIDIA A100 GPU (one of the four available)
*   Annotate frames with detected objects of classes that I chose knowing that I was processing a stream of a CCTV of a road 
*   Start a lightweight HTTP server to stream the annotated video to a chosen port of the compute node 

At startup, the script prints a **ready-to-use SSH port-forwarding command**, similar to:

```text
ssh -N -v -p 8822 -i ~/.ssh/id_ed25519_mlux -L 5000:<compute_node>:5000 user@login.lxp.lu
```

***

## Step 4 — View the live output on your local machine

On **your local machine**, open a new terminal and run the SSH command printed by the script.

Once the tunnel is established, open a browser like Chrome or FireFox and navigate to:

```text
http://localhost:5000/video
```

You should now see the **live YOLO inference output**, streamed directly from the MeluXina compute node.

***

## How the inference script works (briefly)

The `run_inference.py` script follows this logic:

1.  **Stream resolution**
    Uses `yt-dlp` to resolve a YouTube URL into a direct video stream that Yolo can treat 

2.  **Video capture**
    Opens the stream with OpenCV using the FFmpeg backend.

3.  **Model inference**
    Loads a YOLO model from Ultralytics and runs object detection on each frame.

4.  **Frame annotation**
    Draws bounding boxes and class labels onto the video frames.

5.  **Streaming output**
    Encodes frames as JPEG images and serves them as an MJPEG stream via a Flask HTTP server.

This design avoids any dependency on X11 or graphical desktops 



### Overview

The workflow consists of the following steps:

1.  Connect to the **MeluXina** system
2.  Allocate a GPU job
3.  Set up the software environment
4.  Run the YOLO inference python script
5.  Forward a network port to view the live results on your local machine


## Copyright

© 2026 LuxProvide S.A.

This software is the property of **LuxProvide S.A.** and was developed by
**Marco Magliulo**.

Permission is hereby granted to use, modify, and redistribute this code,
in part or in full, for any purpose, without prior permission, provided
that this copyright notice and authorship attribution are retained.

