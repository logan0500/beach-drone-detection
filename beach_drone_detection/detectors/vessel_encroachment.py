"""Detects boats/jet skis entering a swimmer-only zone.

Reuses the swimmer-multiclass model's "boat"/"jetski" classes — no separate
model needed. Geofencing logic: a vessel is flagged as encroaching only if
its center falls inside swim_zone_polygon (a list of (x, y) pixel vertices).
Until a real zone is defined (configs/default.yaml), swim_zone_polygon is
None and this detector just returns raw vessel detections with no
encroachment flag.
"""

from beach_drone_detection.detectors.yolo_base import YoloDetector
from beach_drone_detection.utils.geometry import box_center, point_in_polygon


class VesselEncroachmentDetector(YoloDetector):
    name = "vessel_encroachment"
    model_path = "models/swimmer-multiclass-best.pt"
    conf_threshold = 0.35
    class_filter = {"boat", "jetski"}
    swim_zone_polygon: list[tuple[float, float]] | None = None

    def detect(self, frame):
        vessels = super().detect(frame)
        if not self.swim_zone_polygon:
            return vessels

        for detection in vessels:
            inside = point_in_polygon(box_center(detection.bbox), self.swim_zone_polygon)
            detection.extra["in_swim_zone"] = inside
            if inside:
                detection.label = "vessel_encroachment"
                detection.extra["priority"] = "high"

        return vessels
