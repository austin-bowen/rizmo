from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import count
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


class ImageStore(ABC):
    @abstractmethod
    def add(self, img: np.ndarray, name: str) -> None:
        ...

    @abstractmethod
    def get_all(self) -> Iterable[tuple[np.ndarray, str]]:
        ...


class FileImageStore(ImageStore, ABC):
    def __init__(
            self,
            root: str | Path,
            file_type: str = 'jpg',
            imwrite_params: Sequence[int] = None,
    ):
        self.root = Path(root)
        self.file_type = file_type
        self.imwrite_params = imwrite_params or []

        self.root.mkdir(parents=True, exist_ok=True)


class SingleImagePerNameFileStore(FileImageStore):
    """Stores a single image per name in the file system."""

    def add(self, img: np.ndarray, name: str) -> None:
        path = self.root / f'{name}.{self.file_type}'
        cv2.imwrite(str(path), img, self.imwrite_params)

    def get_all(self) -> Iterable[tuple[np.ndarray, str]]:
        for path in self.root.glob(f'*.{self.file_type}'):
            img = cv2.imread(str(path))
            if img is None:
                print(f'ERROR: Could not read image: {path}')
                continue

            name = path.stem
            yield img, name


class MultiImagePerNameFileStore(FileImageStore):
    """Stores multiple images per name in the file system."""

    def add(self, img: np.ndarray, name: str) -> None:
        path = self._get_next_image_path(name)
        cv2.imwrite(str(path), img, self.imwrite_params)

    def _get_next_image_path(self, name: str) -> Path:
        root = self.root / name
        root.mkdir(parents=True, exist_ok=True)

        for i in count():
            path = root / f'{i}.{self.file_type}'
            if not path.exists():
                return path

        raise RuntimeError('Unreachable')

    def get_all(self) -> Iterable[tuple[np.ndarray, str]]:
        for path in self.root.glob(f'*/*.{self.file_type}'):
            img = cv2.imread(str(path))
            if img is None:
                print(f'ERROR: Could not read image: {path}')
                continue

            name = path.parent.stem
            yield img, name
