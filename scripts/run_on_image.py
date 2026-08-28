#!/usr/bin/env python3
"""Run the configured detectors on a single image and save an annotated copy.

Usage:
    python scripts/run_on_image.py path/to/frame.jpg [-o output.jpg] [-c configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beach_drone_detection.pipeline.config import build_detectors, load_config
from beach_drone_detection.pipeline.runner import Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("-o", "--output", default="output_annotated.jpg")
    parser.add_argument("-c", "--config", default="configs/default.yaml")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    config = load_config(args.config)
    detectors = build_detectors(config)
    pipeline = Pipeline(detectors, config.get("display", {}).get("colors", {}))

    detections_by_detector, annotated, summary = pipeline.run_on_frame(frame)
    cv2.imwrite(args.output, annotated)

    total_detections = sum(len(d) for d in detections_by_detector.values())
    print(f"Ran {len(detectors)} detectors on {args.image}")
    print(f"Total detections: {total_detections}")
    for detector_name, detections in detections_by_detector.items():
        for d in detections:
            print(f"  - [{detector_name}] {d.label}: confidence={d.confidence}, box={d.bbox}")

    print(f"\nSummary: {summary}")
    if summary["drowning_alerts"] > 0:
        print(
            f"\n{summary['drowning_alerts']} DROWNING ALERT(S) - unvalidated model, "
            f"confirm with human review before acting"
        )

    print(f"\nAnnotated output saved to: {args.output}")


if __name__ == "__main__":
    main()
