"""Detects rip currents from top-down video (texture/color pattern analysis)."""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class RipCurrentDetector(BaseDetector):
    name = "rip_current"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
