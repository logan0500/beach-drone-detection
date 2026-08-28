"""Flags likely algal-bloom regions using classical HSV color thresholding.

No trained model backs this — Roboflow datasets for algal blooms are thin/
inconsistent, so this uses a color heuristic instead: convert to HSV,
threshold for the green/turquoise range cyanobacteria scum tends to sit in,
and flag contiguous regions above a minimum size.

This is a first pass, not a validated detector. The HSV range and area
threshold are untuned placeholders — expect false positives on sea foam,
glare, or ordinary shallow-water color variation until this is checked
against real bloom footage and adjusted.
"""

import cv2
import numpy as np

from beach_drone_detection.detectors.base import BaseDetector, Detection

# HSV ranges for common bloom colors (OpenCV convention: H 0-179, S/V 0-255).
DEFAULT_HSV_RANGES = [
    ((35, 60, 60), (85, 255, 255)),  # green / green-yellow scum
]

DEFAULT_MIN_AREA_PX = 400


class WaterQualityDetector(BaseDetector):
    name = "water_quality"

    def __init__(self, hsv_ranges=None, min_area_px=DEFAULT_MIN_AREA_PX):
        self.hsv_ranges = hsv_ranges or DEFAULT_HSV_RANGES
        self.min_area_px = min_area_px

    def detect(self, frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in self.hsv_ranges:
            mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            region_mask = mask[y : y + h, x : x + w]
            coverage = float(np.count_nonzero(region_mask)) / (w * h)
            detections.append(
                Detection(
                    label="algal_bloom",
                    confidence=round(coverage, 3),
                    bbox=(x, y, x + w, y + h),
                    extra={"area_px": int(area)},
                )
            )
        return detections
