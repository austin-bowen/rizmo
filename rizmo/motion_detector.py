from abc import ABC, abstractmethod

import cv2
import numpy as np
from easymesh.utils import require

Image = np.ndarray


class MotionDetector(ABC):
    @abstractmethod
    def is_motion(self, image: Image) -> bool:
        ...


class PixelChangeMotionDetector(MotionDetector):
    def __init__(self, threshold: float, subsample: int = 1):
        require(0 <= threshold <= 1, f'Threshold must be between 0 and 1; got {threshold}')
        require(subsample >= 1, f'Subsample must be at least 1; got {subsample}')

        self.threshold = threshold
        self.subsample = subsample
        self.prev_image = None
        self.diff = 0.

    def is_motion(self, image: Image) -> bool:
        image = image[::self.subsample, ::self.subsample]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_image is None:
            self.prev_image = image
            return False

        self.diff = np.abs(image.astype(float) - self.prev_image)
        self.diff = float(np.mean(self.diff)) / 255

        self.prev_image = image

        return self.diff > self.threshold


class PixelChangeMotionDetectorDynamicThreshold(MotionDetector):
    def __init__(self, change: float, subsample: int = 1, alpha: float = 0.1):
        require(0 <= change <= 1, f'Change must be between 0 and 1; got {change}')
        require(subsample >= 1, f'Subsample must be at least 1; got {subsample}')

        self.change = change
        self.subsample = subsample
        self.alpha = alpha

        self.prev_image = None
        self.diff = 0.
        self.avg_diff = 0.

    def is_motion(self, image: Image) -> bool:
        image = image[::self.subsample, ::self.subsample]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.prev_image is None:
            self.prev_image = image
            return False

        self.diff = np.abs(image.astype(float) - self.prev_image)
        self.diff = float(np.mean(self.diff)) / 255

        is_motion = self.diff / self.avg_diff > self.change

        self.prev_image = image
        self.avg_diff = self.alpha * self.diff + (1 - self.alpha) * self.avg_diff

        return is_motion
