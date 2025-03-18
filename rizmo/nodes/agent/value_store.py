from pathlib import Path

from rizmo.json import JsonFile


class ValueStore:
    def __init__(self, file_path: Path):
        self._json_file = JsonFile(file_path)

        try:
            self.values = self._json_file.read()
        except FileNotFoundError as e:
            print(f'ERROR: {e!r}')
            print(f'Creating new file at "{file_path}".')
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.values = []

    def list(self) -> list[str]:
        return self.values

    def add(self, value: str) -> None:
        self.values.append(value)
        self.save()

    def remove(self, value: str) -> None:
        self.values.remove(value)
        self.save()

    def clear(self) -> None:
        self.values.clear()
        self.save()

    def save(self) -> None:
        self._json_file.write(self.values)
