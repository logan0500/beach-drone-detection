#!/usr/bin/env python3
"""Operator dashboard: shows the drone's camera feed with hazard detections
overlaid, a hazard alert panel with danger level + disclaimer, and manual
control buttons.

No drone is connected yet. The video source is a webcam or a looped video
file standing in for the drone's camera, and manual control just logs the
command and reports back "simulated" - this is where a real flight-control
call (DJI SDK / MAVLink) plugs in once actual hardware exists.

Usage:
    python webapp/app.py --source 0                    # webcam
    python webapp/app.py --source path/to/flight.mp4    # loop a video file
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beach_drone_detection.pipeline.config import build_detectors, load_config
from beach_drone_detection.pipeline.hazards import alerts_from_detections
from beach_drone_detection.pipeline.runner import Pipeline

app = Flask(__name__)

_state_lock = threading.Lock()
state = {"frame_jpeg": None, "alerts": [], "last_command": None, "source_error": None}


def capture_loop(source, config):
    detectors = build_detectors(config)
    pipeline = Pipeline(detectors, config.get("display", {}).get("colors", {}))

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        with _state_lock:
            state["source_error"] = f"Could not open video source: {source}"
        return

    loop_file = isinstance(source, str)

    while True:
        ok, frame = capture.read()
        if not ok:
            if loop_file:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        detections_by_detector, annotated, _ = pipeline.run_on_frame(frame)
        alerts = alerts_from_detections(detections_by_detector)

        ok, jpeg = cv2.imencode(".jpg", annotated)
        if ok:
            with _state_lock:
                state["frame_jpeg"] = jpeg.tobytes()
                state["alerts"] = alerts

        time.sleep(0.03)  # ~30fps processing cap


def mjpeg_generator():
    while True:
        with _state_lock:
            frame = state["frame_jpeg"]
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alerts")
def alerts():
    with _state_lock:
        return jsonify({"alerts": state["alerts"], "source_error": state["source_error"]})


@app.route("/control", methods=["POST"])
def control():
    payload = request.get_json(silent=True) or {}
    command = payload.get("command")
    with _state_lock:
        state["last_command"] = command
    # No drone connected yet - a real flight-control call (DJI SDK / MAVLink)
    # replaces this stub once hardware exists.
    return jsonify(
        {"status": "simulated", "command": command, "note": "No drone connected - command logged only."}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to a video file to loop")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    source = int(args.source) if args.source.isdigit() else args.source

    thread = threading.Thread(target=capture_loop, args=(source, config), daemon=True)
    thread.start()

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
