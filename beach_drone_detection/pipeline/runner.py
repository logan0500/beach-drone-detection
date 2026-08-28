"""Runs the configured set of detectors over a frame and summarizes results."""

from beach_drone_detection.pipeline.annotate import annotate_frame


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
        isolated = detections_by_detector.get("isolated_swimmer", [])
        drowning = [
            d for d in detections_by_detector.get("swimmer_distress", [])
            if d.label == "drowning"
        ]
        encroaching = [
            d for d in detections_by_detector.get("vessel_encroachment", [])
            if d.extra.get("in_swim_zone")
        ]
        return {
            "isolated_swimmers": len(isolated),
            "drowning_alerts": len(drowning),
            "vessel_encroachments": len(encroaching),
        }
