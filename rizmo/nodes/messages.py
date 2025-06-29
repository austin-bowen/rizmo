from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

from rizmo.nodes.messages_py36 import Box


@dataclass
class MotorSystemCommand:
    enabled: bool


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

    speed_dps: float = None

    @property
    def pan_us(self) -> float:
        return 0. if self.pan_deg == 'off' else 1500 + self.pan_deg * US_PER_DEG[0]

    @property
    def tilt0_us(self) -> float:
        return 0. if self.tilt0_deg == 'off' else 1500 + self.tilt0_deg * US_PER_DEG[1]

    @property
    def tilt1_us(self) -> float:
        return 0. if self.tilt1_deg == 'off' else 1500 - self.tilt1_deg * US_PER_DEG[2]

    @property
    def pan_speed_us_per_second(self) -> float:
        return self.speed_dps * US_PER_DEG[0]

    @property
    def tilt0_speed_us_per_second(self) -> float:
        return self.speed_dps * US_PER_DEG[1]

    @property
    def tilt1_speed_us_per_second(self) -> float:
        return self.speed_dps * US_PER_DEG[2]


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


@dataclass
class SetHeadSpeed:
    pan_dps: float = None
    tilt_dps: float = None
    lean_dps: float = None

    @property
    def pan_speed_us_per_second(self) -> float:
        return self.pan_dps * US_PER_DEG[0]

    @property
    def tilt_speed_us_per_second(self) -> float:
        return self.tilt_dps * US_PER_DEG[2]

    @property
    def lean_speed_us_per_second(self) -> float:
        return self.lean_dps * US_PER_DEG[1]


@dataclass
class FaceDetections:
    timestamp: float
    """Timestamp of when the image was taken."""

    image_size: tuple[int, int]
    """Width and height of the image."""

    faces: list['FaceDetection']


@dataclass
class FaceDetection:
    image: np.ndarray
    confidence: float
    box: Box


@dataclass
class FaceRecognitions:
    timestamp: float
    """Timestamp of when the image was taken."""

    image_size: tuple[int, int]
    """Width and height of the image."""

    faces: list['FaceRecognition']


@dataclass
class FaceRecognition:
    name: str | None
    confidence: float
    box: Box
