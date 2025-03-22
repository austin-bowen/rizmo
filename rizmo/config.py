import socket
from dataclasses import dataclass, field
from pathlib import Path

from easymesh.coordinator.constants import DEFAULT_COORDINATOR_PORT
from easymesh.types import Endpoint

IS_RIZMO = socket.gethostname() == 'rizmo'


@dataclass
class Config:
    mesh_coordinator: Endpoint = field(default_factory=lambda: Endpoint(
        host='rizmo.local',
        port=DEFAULT_COORDINATOR_PORT,
    ))

    mesh_authkey: bytes = b'rizmo'

    camera_index: int = 0
    camera_resolution: tuple[int, int] = (1280, 720)

    speaker_device: str = 'M0'
    speaker_mixer: str = 'PCM'
    microphone_mixer: str = 'Headset'
    mic_sample_rate: int = 16000
    mic_block_size: int = 4096

    weather_location: str = 'Anderson, SC'

    memory_file_path: Path = Path('var/memories.json')
    reminders_file_path: Path = Path('var/reminders.json')

    aws_region: str = 'us-east-1'

    wifi_device: str = 'wlan0'


config = Config()
