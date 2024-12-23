"""Python 3.6 compatible message types."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box


@dataclass
class Detections:
    timestamp: float
    """Timestamp of when the image was taken."""

    image_size: Tuple[int, int]
    """Width and height of the image."""

    objects: List[Detection]
