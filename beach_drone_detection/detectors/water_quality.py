"""Flags water quality issues such as algal blooms from surface color/texture."""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class WaterQualityDetector(BaseDetector):
    name = "water_quality"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
