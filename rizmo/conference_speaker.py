"""
Setup:
- pip install pyalsaaudio
"""

from dataclasses import dataclass
from typing import Protocol

try:
    import alsaaudio
except ImportError:
    alsaaudio = None

from rizmo.config import config


@dataclass
class ConferenceSpeaker:
    speaker: 'Mixer'
    microphone: 'Mixer'

    @classmethod
    def build(
            cls,
            device_id: str = config.speaker_device,
            speaker_mixer: str = config.speaker_mixer,
            microphone_mixer: str = config.microphone_mixer,
    ) -> 'ConferenceSpeaker':
        if not alsaaudio:
            return cls(speaker=FakeMixer(), microphone=FakeMixer())

        card_index = alsaaudio.cards().index(device_id)

        return cls(
            speaker=alsaaudio.Mixer(control=speaker_mixer, cardindex=card_index),
            microphone=alsaaudio.Mixer(control=microphone_mixer, cardindex=card_index),
        )


class Mixer(Protocol):
    def getvolume(self) -> list[int]:
        ...

    def setvolume(self, volume: int) -> None:
        ...


class FakeMixer(Mixer):
    def __init__(self):
        self._volume = 50

    def getvolume(self) -> list[int]:
        return [self._volume]

    def setvolume(self, volume: int) -> None:
        self._volume = volume
