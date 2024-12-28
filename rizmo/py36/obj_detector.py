"""Object detector using Jetson Nano hardware."""

import sys
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np

from rizmo.nodes.messages_py36 import Box, Detection

try:
    # Only available on the Jetson Nano
    from jetson.inference import detectNet
    from jetson.utils import cudaFromNumpy
except ImportError:
    detectNet = None
    cudaFromNumpy = None

Image = np.ndarray
"""Image in BGR format."""


def get_object_detector(network: str, threshold: float) -> 'ObjectDetector':
    print('Loading model...')
    model = detectNet(network, sys.argv, threshold)
    print('Done.')

    return DetectNetObjectDetector(model)


class ObjectDetector(ABC):
    @abstractmethod
    def get_objects(self, image) -> List[Detection]:
        ...


class DetectNetObjectDetector(ObjectDetector):
    def __init__(self, model: detectNet) -> None:
        self.model = model

    def get_objects(self, image: Image) -> List[Detection]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cudaFromNumpy(image)
        detectnet_detections = self.model.Detect(image)

        return [detectnet_to_rizmo_detection(d) for d in detectnet_detections]


def detectnet_to_rizmo_detection(model: detectNet, det: detectNet.Detection) -> Detection:
    return Detection(
        label=model.GetClassDesc(det.ClassID),
        confidence=det.Confidence,
        box=Box(
            x=det.Left,
            y=det.Top,
            width=det.Width,
            height=det.Height,
        ),
    )
