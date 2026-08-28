"""Loads each YOLO weights file once and hands the same instance to every
detector that references it (several detectors share swimmer-multiclass-best.pt)."""

_MODEL_CACHE: dict[str, object] = {}


def get_model(path: str):
    if path not in _MODEL_CACHE:
        from ultralytics import YOLO

        _MODEL_CACHE[path] = YOLO(path)
    return _MODEL_CACHE[path]
