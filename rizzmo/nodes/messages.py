from dataclasses import dataclass


@dataclass(slots=True)
class Box:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box: Box
