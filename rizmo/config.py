import socket
from dataclasses import dataclass, field

from easymesh.coordinator.constants import DEFAULT_COORDINATOR_PORT
from easymesh.types import Endpoint

IS_RIZMO = socket.gethostname() == 'rizmo'


@dataclass
class Config:
    coordinator: Endpoint = field(default_factory=lambda: Endpoint(
        host='rizmo.local',
        port=DEFAULT_COORDINATOR_PORT,
    ))

    camera_index: int = 0


config = Config()
