import json
from pathlib import Path
from typing import Any, Union

PathOrStr = Union[Path, str]


class JsonFile:
    """Makes reading and writing JSON files easier."""

    file_path: Path

    def __init__(
            self,
            file_path: PathOrStr,
            indent: int = 2,
    ):
        self.file_path = Path(file_path)
        self.indent = indent

    def read(self) -> Any:
        with self.file_path.open('r') as file:
            return json.load(file)

    def write(self, data: Any) -> None:
        with self.file_path.open('w') as file:
            json.dump(data, file, indent=self.indent)
