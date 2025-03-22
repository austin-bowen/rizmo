"""
Setup:
- pip install pyalsaaudio
"""

from dataclasses import dataclass

import alsaaudio

from rizmo.config import config


@dataclass
class ConferenceSpeaker:
    speaker: alsaaudio.Mixer
    microphone: alsaaudio.Mixer

    @classmethod
    def build(
            cls,
            device_id: str = config.speaker_device,
            speaker_mixer: str = config.speaker_mixer,
            microphone_mixer: str = config.microphone_mixer,
    ) -> 'ConferenceSpeaker':
        card_index = alsaaudio.cards().index(device_id)

        return cls(
            speaker=alsaaudio.Mixer(control=speaker_mixer, cardindex=card_index),
            microphone=alsaaudio.Mixer(control=microphone_mixer, cardindex=card_index),
        )
