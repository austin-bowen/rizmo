from abc import ABC, abstractmethod

import cv2
import numpy as np
from easymesh.utils import require

Image = np.ndarray


class MotionDetector(ABC):
    @abstractmethod
    def is_motion(self, image: Image) -> bool:
        ...


class DynamicThresholdPixelChangeMotionDetector(MotionDetector):
    def __init__(self, change: float = 0.1, alpha: float = 0.01, subsample: int = 1):
        require(0 <= change <= 1, f'Change must be between 0 and 1; got {change}')
        require(0 <= alpha <= 1, f'Alpha must be between 0 and 1; got {alpha}')
        require(subsample >= 1, f'Subsample must be at least 1; got {subsample}')

        self.change = change
        self.alpha = alpha
        self.subsample = subsample

        self.prev_image = None
        self.diff = 0.
        self.avg_diff = float('inf')

    def is_motion(self, image: Image) -> bool:
        image = image[::self.subsample, ::self.subsample]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_image is None:
            self.prev_image = image
            return False

        self.diff = np.abs(image.astype(np.int16) - self.prev_image)
        self.diff = float(np.mean(self.diff)) / 255

        self.avg_diff = min(self.avg_diff, self.diff)

        change = (self.diff - self.avg_diff) / (self.avg_diff + 1e-6)
        is_motion = change > self.change

        self.prev_image = image
        self.avg_diff = self.alpha * self.diff + (1 - self.alpha) * self.avg_diff

        return is_motion
