from abc import ABC, abstractmethod
from collections.abc import Sized
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn.functional import normalize

Label = Any
Vector = np.ndarray


class VectorDatabase(Sized, ABC):
    @abstractmethod
    def add_all(self, items: Iterable[tuple[Vector, Label]]) -> None:
        ...

    @abstractmethod
    def search(self, vectors: Iterable[Vector]) -> list[tuple[Label, float]]:
        ...


class BruteForceCosineVectorDatabase(VectorDatabase):
    def __init__(
            self,
            dtype: torch.dtype = None,
            device: str = None,
    ):
        self.dtype = dtype
        self.device = device

        self.vectors: torch.Tensor | None = None
        self.labels: list[Label] = []

    def __len__(self) -> int:
        return len(self.labels)

    def add_all(self, items: Iterable[tuple[Vector, Label]]) -> None:
        vectors, labels = zip(*items)

        new_vectors = self._normalize(vectors)

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = torch.cat([self.vectors, new_vectors], dim=0)

        self.labels.extend(labels)

        assert self.vectors.size(0) == len(self.labels)

    def search(self, vectors: Iterable[Vector]) -> list[tuple[Label, float]]:
        if self.vectors is None:
            raise ValueError("No vectors in the database.")

        similarities = self.get_similarities(vectors)

        similarities, most_similar_indices = torch.max(similarities, dim=1)
        similarities = similarities.cpu().numpy()
        most_similar_indices = most_similar_indices.cpu().numpy()

        return [
            (self.labels[label_i], float(sim))
            for label_i, sim in zip(most_similar_indices, similarities)
        ]

    def get_similarities(self, vectors: Iterable[Vector]) -> torch.Tensor:
        embeddings = self._normalize(vectors)
        return embeddings @ self.vectors.T

    def _normalize(self, vectors: Iterable[Vector]) -> torch.Tensor:
        vectors = np.stack(list(vectors), axis=0)
        vectors = torch.tensor(vectors, dtype=self.dtype, device=self.device)
        normalize(vectors, dim=1, out=vectors)
        return vectors
