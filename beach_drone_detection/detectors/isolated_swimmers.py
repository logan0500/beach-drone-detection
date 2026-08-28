"""Flags swimmers who are far from any group (higher risk if something goes wrong).

Derived logic on top of the swimmer-multiclass model's "swimmer" class: pull
every swimmer detection in the frame, compute pairwise center-to-center
distances, and flag anyone whose nearest neighbor is farther than
isolation_distance_px away. That threshold is a placeholder (150px) until
real drone altitude/camera specs are known — see configs/default.yaml.
"""

from beach_drone_detection.detectors.yolo_base import YoloDetector
from beach_drone_detection.utils.geometry import box_center, euclidean_distance


class IsolatedSwimmerDetector(YoloDetector):
    name = "isolated_swimmer"
    model_path = "models/swimmer-multiclass-best.pt"
    conf_threshold = 0.35
    class_filter = {"swimmer"}
    isolation_distance_px = 150

    def detect(self, frame):
        swimmers = super().detect(frame)
        centers = [box_center(d.bbox) for d in swimmers]

        isolated = []
        for i, (detection, center) in enumerate(zip(swimmers, centers)):
            others = [c for j, c in enumerate(centers) if j != i]
            if others:
                nearest_dist = min(euclidean_distance(center, o) for o in others)
                is_isolated = nearest_dist > self.isolation_distance_px
            else:
                nearest_dist = None
                is_isolated = True  # only swimmer detected -> isolated

            if is_isolated:
                detection.label = "isolated_swimmer"
                detection.extra["nearest_neighbor_distance_px"] = nearest_dist
                isolated.append(detection)

        return isolated
