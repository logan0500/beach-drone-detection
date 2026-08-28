"""Flags water quality issues such as algal blooms from surface color/texture.

Not started: no trained model exists (Roboflow datasets for this are thin/
inconsistent). Planned approach is classical CV rather than a YOLO model —
convert frames to HSV and threshold for known bloom-color ranges — since a
labeled dataset isn't available to train on.
"""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class WaterQualityDetector(BaseDetector):
    name = "water_quality"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
