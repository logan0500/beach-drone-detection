"""Loads a config YAML file and builds the detector set it describes."""

import yaml

from beach_drone_detection.detectors.isolated_swimmers import IsolatedSwimmerDetector
from beach_drone_detection.detectors.marine_life import MarineLifeDetector
from beach_drone_detection.detectors.rip_currents import RipCurrentDetector
from beach_drone_detection.detectors.swimmer_distress import SwimmerDistressDetector
from beach_drone_detection.detectors.vessel_encroachment import VesselEncroachmentDetector

DETECTOR_CLASSES = {
    "rip_current": RipCurrentDetector,
    "marine_life": MarineLifeDetector,
    "swimmer_distress": SwimmerDistressDetector,
    "isolated_swimmer": IsolatedSwimmerDetector,
    "vessel_encroachment": VesselEncroachmentDetector,
    # "water_quality" has no working detector yet - omitted on purpose.
}


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

        detectors[name] = detector
    return detectors
