"""Detects sharks and other marine life from top-down video."""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class MarineLifeDetector(BaseDetector):
    name = "marine_life"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
