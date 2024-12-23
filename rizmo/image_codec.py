from abc import ABC, abstractmethod

import cv2
import numpy as np

from easymesh.utils import require

Image = np.ndarray


class ImageCodec(ABC):
    @abstractmethod
    def encode(self, image: Image) -> bytes:
        ...

    @abstractmethod
    def decode(self, data: bytes) -> Image:
        ...


class JpegImageCodec(ImageCodec):
    def __init__(self, quality: int = 90):
        """
        Args:
            quality: Int in range 0-100.
        """

        require(0 <= quality <= 100, f'Quality must be in range 0-100; got {quality}')

        self.quality = quality

    def encode(self, image: Image) -> bytes:
        _, encoded = cv2.imencode(
            '.jpg',
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )

        return encoded.tobytes()

    def decode(self, data: bytes) -> Image:
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
