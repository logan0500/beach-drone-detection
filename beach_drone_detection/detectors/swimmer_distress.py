"""Detects swimmers in apparent distress ("drowning" vs. "swimming" pose).

NOTE: this model is trained but not yet independently validated against real
test footage. "drowning" detections are marked as needing human confirmation
rather than treated as an automatic alert until that validation happens.
"""

from beach_drone_detection.detectors.yolo_base import YoloDetector


class SwimmerDistressDetector(YoloDetector):
    name = "swimmer_distress"
    model_path = "models/swimmer-distress-best.pt"
    conf_threshold = 0.4

    def detect(self, frame):
        detections = super().detect(frame)
        for detection in detections:
            if detection.label == "drowning":
                detection.extra["priority"] = "high"
                detection.extra["needs_human_confirmation"] = True
        return detections
