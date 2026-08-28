"""Detects rip currents from top-down video.

The source Roboflow dataset never named its single class, so the model
reports it as "0" rather than "rip_current" — cosmetic, not a bug.
"""

from beach_drone_detection.detectors.yolo_base import YoloDetector


class RipCurrentDetector(YoloDetector):
    name = "rip_current"
    model_path = "models/rip-best.pt"
    conf_threshold = 0.4
