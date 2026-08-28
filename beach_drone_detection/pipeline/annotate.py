"""Draws detector output onto a frame for human review."""

import cv2

DEFAULT_COLOR = (200, 200, 200)  # gray, BGR


def _color_for(detector_name, label, colors_config):
    entry = colors_config.get(detector_name)
    if isinstance(entry, dict):
        color = entry.get(label, DEFAULT_COLOR)
    elif entry is not None:
        color = entry
    else:
        color = DEFAULT_COLOR
    return tuple(color)


def annotate_frame(frame, detections_by_detector, colors_config=None):
    """detections_by_detector: {detector_name: [Detection, ...]}"""
    colors_config = colors_config or {}
    annotated = frame.copy()

    for detector_name, detections in detections_by_detector.items():
        for detection in detections:
            if detection.bbox is None:
                continue
            x1, y1, x2, y2 = detection.bbox
            color = _color_for(detector_name, detection.label, colors_config)
            label_text = f"{detection.label} {detection.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated, label_text, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )

            if detector_name == "isolated_swimmer":
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 20, color, 3)

    return annotated
