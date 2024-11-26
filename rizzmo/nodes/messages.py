from dataclasses import dataclass
from typing import Literal, Union


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


@dataclass(slots=True)
class Detections:
    timestamp: float
    objects: list[Detection]


ServoPositionDeg = Union[float, Literal['off'], None]

US_PER_DEG = (
    500 / 72.3,
    500 / 45,
    500 / 45,
)


@dataclass
class SetServoPosition:
    pan_deg: ServoPositionDeg = None
    tilt0_deg: ServoPositionDeg = None
    tilt1_deg: ServoPositionDeg = None

    @property
    def pan_us(self) -> float:
        return 0. if self.pan_deg == 'off' else 1500 + self.pan_deg * US_PER_DEG[0]

    @property
    def tilt0_us(self) -> float:
        return 0. if self.tilt0_deg == 'off' else 1500 + self.tilt0_deg * US_PER_DEG[1]

    @property
    def tilt1_us(self) -> float:
        return 0. if self.tilt1_deg == 'off' else 1500 - self.tilt1_deg * US_PER_DEG[2]


@dataclass
class ChangeServoPosition:
    pan_deg: float = None
    tilt0_deg: float = None
    tilt1_deg: float = None

    @property
    def pan_us(self) -> float:
        return self.pan_deg * US_PER_DEG[0]

    @property
    def tilt0_us(self) -> float:
        return self.tilt0_deg * US_PER_DEG[1]

    @property
    def tilt1_us(self) -> float:
        return self.tilt1_deg * US_PER_DEG[2]
