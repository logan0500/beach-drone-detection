"""Loads a config YAML file and builds the detector set it describes."""

import yaml

from beach_drone_detection.detectors.isolated_swimmers import IsolatedSwimmerDetector
from beach_drone_detection.detectors.marine_life import MarineLifeDetector
from beach_drone_detection.detectors.rip_currents import RipCurrentDetector
from beach_drone_detection.detectors.swimmer_distress import SwimmerDistressDetector
from beach_drone_detection.detectors.swimmer_distress_motion import SwimmerDistressMotionDetector
from beach_drone_detection.detectors.vessel_encroachment import VesselEncroachmentDetector
from beach_drone_detection.detectors.water_quality import WaterQualityDetector

DETECTOR_CLASSES = {
    "rip_current": RipCurrentDetector,
    "marine_life": MarineLifeDetector,
    "swimmer_distress": SwimmerDistressDetector,
    "swimmer_distress_motion": SwimmerDistressMotionDetector,
    "isolated_swimmer": IsolatedSwimmerDetector,
    "vessel_encroachment": VesselEncroachmentDetector,
    "water_quality": WaterQualityDetector,
}

_MOTION_TUNABLE_KEYS = (
    "match_distance_px", "max_missed_frames", "stillness_window",
    "stillness_threshold_px", "active_movement_px", "erratic_window",
    "erratic_reversal_count",
)


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_detectors(config: dict) -> dict:
    detectors = {}
    for name, cfg in config.get("detectors", {}).items():
        if not cfg.get("enabled"):
            continue
        detector_cls = DETECTOR_CLASSES.get(name)
        if detector_cls is None:
            continue

        kwargs = {}
        if "model_path" in cfg:
            kwargs["model_path"] = cfg["model_path"]
        if "conf_threshold" in cfg:
            kwargs["conf_threshold"] = cfg["conf_threshold"]

        detector = detector_cls(**kwargs)

        if name == "isolated_swimmer" and "isolation_distance_px" in cfg:
            detector.isolation_distance_px = cfg["isolation_distance_px"]
        if name == "vessel_encroachment" and cfg.get("swim_zone_polygon"):
            detector.swim_zone_polygon = cfg["swim_zone_polygon"]
        if name == "water_quality":
            if cfg.get("hsv_ranges"):
                detector.hsv_ranges = [tuple(map(tuple, r)) for r in cfg["hsv_ranges"]]
            if "min_area_px" in cfg:
                detector.min_area_px = cfg["min_area_px"]
        if name == "swimmer_distress_motion":
            for key in _MOTION_TUNABLE_KEYS:
                if key in cfg:
                    setattr(detector, key, cfg[key])

        detectors[name] = detector
    return detectors
