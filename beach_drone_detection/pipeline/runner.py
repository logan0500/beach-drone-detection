"""Runs the configured set of detectors over a frame and summarizes results."""

from beach_drone_detection.pipeline.annotate import annotate_frame
from beach_drone_detection.pipeline.hazards import alerts_from_detections


class Pipeline:
    def __init__(self, detectors: dict, colors_config: dict | None = None):
        self.detectors = detectors
        self.colors_config = colors_config or {}

    def run_on_frame(self, frame):
        detections_by_detector = {
            name: detector.detect(frame) for name, detector in self.detectors.items()
        }
        annotated = annotate_frame(frame, detections_by_detector, self.colors_config)
        summary = self._summarize(detections_by_detector)
        return detections_by_detector, annotated, summary

    def _summarize(self, detections_by_detector):
        counts = {}
        for alert in alerts_from_detections(detections_by_detector):
            counts[alert["detector"]] = counts.get(alert["detector"], 0) + 1
        return {
            "isolated_swimmers": counts.get("isolated_swimmer", 0),
            "drowning_alerts": counts.get("swimmer_distress", 0),
            "vessel_encroachments": counts.get("vessel_encroachment", 0),
            "possible_distress_motion_flags": counts.get("swimmer_distress_motion", 0),
        }
