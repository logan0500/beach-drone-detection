#!/usr/bin/env python3
"""Run the configured detectors on a video file, frame by frame, and save an
annotated copy of the video plus a summary of what was seen.

Usage:
    python scripts/run_on_video.py path/to/flight.mp4 [-o output.mp4] [-c configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beach_drone_detection.pipeline.config import build_detectors, load_config
from beach_drone_detection.pipeline.runner import Pipeline
from beach_drone_detection.pipeline.video_source import VideoSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("-o", "--output", default="output_annotated.mp4")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    parser.add_argument(
        "--log-every", type=int, default=30,
        help="Print a progress line every N frames (default: 30)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    detectors = build_detectors(config)
    pipeline = Pipeline(detectors, config.get("display", {}).get("colors", {}))

    source = VideoSource(args.video)
    width, height = source.frame_size
    fps = source.fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    totals = {"isolated_swimmers": 0, "drowning_alerts": 0, "vessel_encroachments": 0}
    frames_with_drowning_alert = 0
    frame_count = 0

    print(f"Running {len(detectors)} detectors on {args.video} ({width}x{height} @ {fps:.1f}fps)")

    for frame in source.frames():
        frame_count += 1
        _, annotated, summary = pipeline.run_on_frame(frame)
        writer.write(annotated)

        for key in totals:
            totals[key] += summary[key]
        if summary["drowning_alerts"] > 0:
            frames_with_drowning_alert += 1

        if frame_count % args.log_every == 0:
            print(f"  frame {frame_count}: {summary}")

    writer.release()

    print(f"\nProcessed {frame_count} frames")
    print(f"Isolated-swimmer detections across video: {totals['isolated_swimmers']}")
    print(f"Vessel-encroachment detections across video: {totals['vessel_encroachments']}")
    print(f"Frames with a drowning alert: {frames_with_drowning_alert}")
    if frames_with_drowning_alert > 0:
        print(
            "\nDROWNING ALERT(S) present - unvalidated model, "
            "confirm with human review before acting"
        )

    print(f"\nAnnotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
