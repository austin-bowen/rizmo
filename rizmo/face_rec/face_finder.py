from abc import ABC, abstractmethod
from typing import Iterable, Literal

import numpy as np
import torch

from rizmo.face_rec.face_embedder import FaceEmbeddingGenerator, InsightFaceEmbeddingGenerator
from rizmo.face_rec.vector_db import BruteForceCosineVectorDatabase, VectorDatabase

ModelName = Literal['w600k_r50', 'w600k_mbf']


def build_face_finder(
        embedding_model: ModelName = 'w600k_r50',
        embedding_model_root: str = './resources/insightface/models',
        face_db_device: str = 'auto',
        min_similarity: float = 0.001,
) -> 'SimilarFaceFinder':
    model_file = f'{embedding_model_root}/{embedding_model}.onnx'
    face_embedder = InsightFaceEmbeddingGenerator.from_model_zoo(model_file)

    if face_db_device == 'auto':
        face_db_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_db = BruteForceCosineVectorDatabase(
        dtype=torch.float16,
        device=face_db_device,
    )

    similar_face_finder = SimilarFaceEmbeddingFinder(
        face_embedder,
        face_db,
        threshold=min_similarity,
    )

    return similar_face_finder


class SimilarFaceFinder(ABC):
    @abstractmethod
    def add_face(self, face: np.ndarray, name: str) -> None:
        ...

    @abstractmethod
    def add_faces(self, faces: Iterable[tuple[np.ndarray, str]]) -> None:
        ...

    @abstractmethod
    def find_faces(self, imgs: Iterable[np.ndarray]) -> list[tuple[str | None, float]]:
        ...


class SimilarFaceEmbeddingFinder(SimilarFaceFinder):
    def __init__(
            self,
            face_embedder: FaceEmbeddingGenerator,
            vector_db: VectorDatabase,
            threshold: float,
            not_found_result: tuple[str | None, float] = (None, 0.0),
    ):
        self.face_embedder = face_embedder
        self.vector_db = vector_db
        self.threshold = threshold
        self.not_found_result = not_found_result

    def add_face(self, face: np.ndarray, name: str) -> None:
        self.add_faces([(face, name)])

    def add_faces(self, faces: Iterable[tuple[np.ndarray, str]]) -> None:
        faces = list(faces)
        if not faces:
            return

        imgs, names = zip(*faces)
        embeddings = self._get_embeddings(imgs)
        self.vector_db.add_all(zip(embeddings, names))

    def find_faces(self, imgs: Iterable[np.ndarray]) -> list[tuple[str | None, float]]:
        if not len(self.vector_db):
            return [self.not_found_result for _ in imgs]

        embeddings = self._get_embeddings(imgs)

        faces: list[tuple[str, float]] = self.vector_db.search(embeddings)

        return [
            face if face[1] >= self.threshold else self.not_found_result
            for face in faces
        ]

    def _get_embeddings(self, imgs: Iterable[np.ndarray]) -> list[np.ndarray]:
        return self.face_embedder.get_embeddings(list(imgs))
