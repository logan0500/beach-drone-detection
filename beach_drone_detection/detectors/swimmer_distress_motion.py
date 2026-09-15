"""Flags "possible distress" using motion patterns across video frames — a
second, complementary signal alongside the trained swimmer_distress model.

swimmer-distress-best.pt classifies a single frame's body pose ("drowning"
vs. "swimming"). This detector looks at something a single frame can't show:
how a swimmer's position changes over time. Two patterns are used as
proxies, both standard heuristics in drowning-safety literature:
  - prolonged stillness: a swimmer who was moving and then goes still for
    an extended stretch (possible submersion/unconsciousness)
  - erratic movement: sharp, repeated direction reversals in a short window
    (thrashing)

This is a heuristic, not a trained or validated model. It has no ground
truth to check against yet, the thresholds below are untuned placeholders,
and it should be treated as a signal to cross-check against
swimmer_distress — not a standalone alert. It only produces signal on
video (it needs several frames of history); a single image gives it
nothing to compare against.

Tracking is a simple greedy nearest-centroid match frame-to-frame, not a
real multi-object tracker — it will misassociate swimmers who cross paths
or get too close together. Good enough for a first pass, not for crowded
scenes.
"""

import math

from beach_drone_detection.detectors.base import Detection
from beach_drone_detection.detectors.yolo_base import YoloDetector
from beach_drone_detection.utils.geometry import box_center, euclidean_distance

MAX_TRACK_HISTORY = 200


def _path_length(points):
    return sum(euclidean_distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def _vector_angle_deg(v1, v2):
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _direction_reversal_count(points, angle_threshold_deg=90, min_step_px=3):
    """Count sharp direction changes between consecutive steps, ignoring any
    step smaller than min_step_px - without that floor, a swimmer holding
    still would register spurious "reversals" from detection-box jitter
    alone (each sub-pixel wobble counts as a new direction)."""
    vectors = []
    for i in range(1, len(points)):
        vec = (points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        if math.hypot(*vec) >= min_step_px:
            vectors.append(vec)

    if len(vectors) < 2:
        return 0
    reversals = 0
    prev_vec = vectors[0]
    for vec in vectors[1:]:
        if _vector_angle_deg(prev_vec, vec) > angle_threshold_deg:
            reversals += 1
        prev_vec = vec
    return reversals


class _Track:
    def __init__(self, center):
        self.history = [center]
        self.frames_since_seen = 0


class SwimmerDistressMotionDetector(YoloDetector):
    name = "swimmer_distress_motion"
    model_path = "models/swimmer-multiclass-best.pt"
    conf_threshold = 0.35
    class_filter = {"swimmer"}

    match_distance_px = 80       # max centroid jump between frames to count as the same swimmer
    max_missed_frames = 5        # drop a track after this many frames with no match

    # Frame counts below assume ~30fps aerial footage - rescale if yours differs.
    stillness_window = 30        # "recent" window length, compared against an equal window before it
    stillness_threshold_px = 15  # total movement below this over the recent window = "still"
    active_movement_px = 40      # movement above this in the earlier window = "was actively swimming"
    erratic_window = 10
    erratic_reversal_count = 4
    erratic_min_step_px = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tracks: list[_Track] = []

    def detect(self, frame):
        swimmers = super().detect(frame)
        centers = [box_center(d.bbox) for d in swimmers]

        candidates = []
        for i, center in enumerate(centers):
            for t, track in enumerate(self._tracks):
                dist = euclidean_distance(track.history[-1], center)
                if dist <= self.match_distance_px:
                    candidates.append((dist, i, t))
        candidates.sort(key=lambda c: c[0])

        detection_to_track = {}
        assigned_tracks = set()
        for dist, i, t in candidates:
            if i in detection_to_track or t in assigned_tracks:
                continue
            detection_to_track[i] = t
            assigned_tracks.add(t)

        distress_detections = []
        matched_track_indices = set()

        for i, (detection, center) in enumerate(zip(swimmers, centers)):
            if i in detection_to_track:
                track_idx = detection_to_track[i]
                track = self._tracks[track_idx]
                track.history.append(center)
                track.history = track.history[-MAX_TRACK_HISTORY:]
                track.frames_since_seen = 0
            else:
                track = _Track(center)
                self._tracks.append(track)
                track_idx = len(self._tracks) - 1
            matched_track_indices.add(track_idx)

            reason = self._check_distress(track)
            if reason:
                distress_detections.append(
                    Detection(
                        label="possible_distress",
                        confidence=1.0,  # heuristic trigger, not a calibrated probability
                        bbox=detection.bbox,
                        extra={"reason": reason, "unvalidated": True},
                    )
                )

        for t, track in enumerate(self._tracks):
            if t not in matched_track_indices:
                track.frames_since_seen += 1
        self._tracks = [t for t in self._tracks if t.frames_since_seen <= self.max_missed_frames]

        return distress_detections

    def _check_distress(self, track):
        history = track.history

        if len(history) >= self.stillness_window * 2:
            earlier = history[-self.stillness_window * 2 : -self.stillness_window]
            recent = history[-self.stillness_window :]
            if _path_length(earlier) > self.active_movement_px and _path_length(recent) < self.stillness_threshold_px:
                return "prolonged_stillness"

        if len(history) >= self.erratic_window:
            recent = history[-self.erratic_window :]
            reversals = _direction_reversal_count(recent, min_step_px=self.erratic_min_step_px)
            if reversals >= self.erratic_reversal_count:
                return "erratic_movement"

        return None
