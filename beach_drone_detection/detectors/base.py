"""Shared interface that every detector below implements."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    """One finding from a detector, e.g. 'rip current at these pixels'."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseDetector:
    """Every detector (rip currents, sharks, swimmers, ...) implements this."""

    name: str = "base"

    def detect(self, frame) -> list[Detection]:
        raise NotImplementedError
