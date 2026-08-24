"""Detects swimmers in apparent distress (e.g. abnormal motion patterns)."""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class SwimmerDistressDetector(BaseDetector):
    name = "swimmer_distress"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
