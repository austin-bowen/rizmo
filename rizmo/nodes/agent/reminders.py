from pathlib import Path

from rizmo.json import JsonFile


class ReminderSystem:
    def __init__(self, file_path: Path):
        self._json_file = JsonFile(file_path)

        try:
            self.reminders = self._json_file.read()
        except FileNotFoundError as e:
            print(f'ERROR: {e!r}')
            print(f'creating new reminders file at "{file_path}".')
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.reminders = []

    def list(self) -> list[str]:
        return self.reminders

    def add(self, reminder: str) -> None:
        self.reminders.append(reminder)
        self.save()

    def remove(self, reminder: str) -> None:
        self.reminders.remove(reminder)
        self.save()

    def clear(self) -> None:
        self.reminders.clear()
        self.save()

    def save(self) -> None:
        self._json_file.write(self.reminders)
