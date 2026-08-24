"""Flags swimmers who are far from any group (higher risk if something goes wrong)."""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class IsolatedSwimmerDetector(BaseDetector):
    name = "isolated_swimmer"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
