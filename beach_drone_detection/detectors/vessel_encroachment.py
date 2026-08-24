"""Detects boats/jet skis/vessels entering swimmer-only zones."""

from beach_drone_detection.detectors.base import BaseDetector, Detection


class VesselEncroachmentDetector(BaseDetector):
    name = "vessel_encroachment"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
