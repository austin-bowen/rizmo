"""Object detector using Jetson Nano hardware."""

import sys
from abc import ABC, abstractmethod

import cv2
import numpy as np

try:
    from jetson.inference import detectNet
except ImportError:
    def detectNet(*args, **kwargs):
        ...

Image = np.ndarray
"""Image in BGR format."""


class ObjectDetector(ABC):
    @abstractmethod
    def get_objects(self, image):
        ...


class DetectNetObjectDetector(ObjectDetector):
    def __init__(self, model: detectNet) -> None:
        self.model = model

    def get_objects(self, image: Image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.model.Detect(image)


def get_object_detector(network: str, threshold: float) -> ObjectDetector:
    print('Loading model...')
    model = detectNet(network, sys.argv, threshold)
    print('Done.')

    return DetectNetObjectDetector(model)
