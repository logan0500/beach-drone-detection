"""Shared logic for detectors backed by a trained YOLO model.

Subclasses set model_path / conf_threshold / class_filter as class attributes
and get a working detect() for free. Detectors that need extra logic on top
(isolated swimmers, drowning alerts) override detect() and call super().
"""

from beach_drone_detection.detectors.base import BaseDetector, Detection
from beach_drone_detection.utils.model_registry import get_model


class YoloDetector(BaseDetector):
    model_path: str = ""
    conf_threshold: float = 0.4
    class_filter: set[str] | None = None  # None = keep every class the model predicts

    def __init__(self, model_path=None, conf_threshold=None, class_filter=None):
        if model_path is not None:
            self.model_path = model_path
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if class_filter is not None:
            self.class_filter = class_filter
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_model(self.model_path)
        return self._model

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_name = self.model.names[int(box.cls[0])]

            if self.class_filter is not None and class_name not in self.class_filter:
                continue

            detections.append(
                Detection(
                    label=class_name,
                    confidence=round(confidence, 3),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
