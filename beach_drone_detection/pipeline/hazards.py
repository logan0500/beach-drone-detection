"""Hazard metadata (display name, danger level, disclaimer) and the logic
for which raw detections actually count as an alert worth surfacing to a
human. Shared by the CLI runner and the web dashboard so both agree on
what "isolated_swimmer" or "vessel_encroachment" actually means.

Danger levels and disclaimers here are placeholders reflecting current
validation status, not a safety certification - update disclaimer text
the moment a detector's real-world accuracy is actually known.
"""

HAZARD_INFO = {
    "marine_life": {
        "display_name": "Shark / Marine Life",
        "danger_level": "critical",
        "disclaimer": "Automated detection from an unvalidated model — confirm visually before acting.",
    },
    "rip_current": {
        "display_name": "Rip Current",
        "danger_level": "high",
        "disclaimer": "Automated detection from an unvalidated model — confirm before acting.",
    },
    "isolated_swimmer": {
        "display_name": "Isolated Swimmer",
        "danger_level": "medium",
        "disclaimer": "Distance-based heuristic on an unvalidated model — flags distance from others, not distress.",
    },
    "vessel_encroachment": {
        "display_name": "Vessel in Swim Zone",
        "danger_level": "high",
        "disclaimer": "Requires a configured swim-zone polygon to be meaningful; unvalidated model.",
    },
    "swimmer_distress": {
        "display_name": "Possible Drowning (pose-based)",
        "danger_level": "critical",
        "disclaimer": "Model is NOT validated — treat as needing immediate human confirmation, not as ground truth.",
    },
    "swimmer_distress_motion": {
        "display_name": "Possible Distress (motion-based)",
        "danger_level": "critical",
        "disclaimer": "Heuristic, not a trained model — cross-check against the pose-based signal and confirm visually.",
    },
}


def _is_alert(detector_name, detection):
    if detector_name == "swimmer_distress":
        return detection.label == "drowning"
    if detector_name == "vessel_encroachment":
        return detection.label == "vessel_encroachment"
    return True  # marine_life, rip_current, isolated_swimmer, swimmer_distress_motion:
    # every detection these produce is already alert-worthy by construction.


def alerts_from_detections(detections_by_detector: dict) -> list[dict]:
    """Flatten {detector_name: [Detection, ...]} into a list of alert dicts
    ready to show a human, filtering out detections that aren't actually
    hazards yet (e.g. "swimming" from swimmer_distress, or a boat outside
    the swim zone)."""
    alerts = []
    for detector_name, detections in detections_by_detector.items():
        info = HAZARD_INFO.get(detector_name)
        if info is None:
            continue
        for detection in detections:
            if not _is_alert(detector_name, detection):
                continue
            alerts.append(
                {
                    "detector": detector_name,
                    "label": detection.label,
                    "display_name": info["display_name"],
                    "danger_level": info["danger_level"],
                    "disclaimer": info["disclaimer"],
                    "confidence": detection.confidence,
                    "bbox": detection.bbox,
                }
            )
    return alerts
