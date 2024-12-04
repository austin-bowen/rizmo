import socket
from dataclasses import dataclass, field

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

    weather_location: str = 'Anderson, SC'


config = Config()
