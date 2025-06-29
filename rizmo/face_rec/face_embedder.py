from abc import ABC, abstractmethod

import cv2
import insightface
import numpy as np


class FaceEmbeddingGenerator(ABC):
    @abstractmethod
    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        ...

    def get_embeddings(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        return [self.get_embedding(img) for img in imgs]


class InsightFaceEmbeddingGenerator(FaceEmbeddingGenerator):
    def __init__(self, model, img_size: int):
        self.model = model
        self.img_size = img_size

    @classmethod
    def from_model_zoo(
            cls,
            name: str,
            img_size: int = 112,
            ctx_id: int = 0,
            prepare_kwargs: dict = None,
            **kwargs,
    ) -> 'InsightFaceEmbeddingGenerator':
        prepare_kwargs = prepare_kwargs or {}
        model = insightface.model_zoo.get_model(name, **kwargs)
        model.prepare(ctx_id=ctx_id, **prepare_kwargs)
        return cls(model, img_size)

    def get_embedding(self, img: np.ndarray) -> np.ndarray:
        return self.get_embeddings([img])[0]

    def get_embeddings(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        imgs = [preprocess_face(img, size=self.img_size) for img in imgs]
        return self.model.get_feat(imgs)


def preprocess_face(face_img: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Resizes the input face image to 112x112 pixels and applies padding if necessary.
    """

    assert size % 2 == 0, f'Size must be an even number; got {size}'

    h, w = face_img.shape[:2]
    scale = .5 * size / max(h, w)
    new_w, new_h = 2 * int(w * scale), 2 * int(h * scale)

    resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, size - new_h - pad_h,
        pad_w, size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    assert padded.shape[:2] == (size, size), \
        f"Expected output shape {(size, size)}, got {padded.shape[:2]}"

    return padded
