"""Detects sharks and other marine life from top-down video."""

from beach_drone_detection.detectors.yolo_base import YoloDetector


class MarineLifeDetector(YoloDetector):
    name = "marine_life"
    model_path = "models/shark-best.pt"
    conf_threshold = 0.4
