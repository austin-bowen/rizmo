import socket
from dataclasses import dataclass

from easymesh.coordinator.constants import DEFAULT_COORDINATOR_PORT
from easymesh.types import Endpoint

IS_RIZZMO = socket.gethostname() == 'rizzmo'


@dataclass
class Config:
    coordinator: Endpoint = Endpoint(
        host='rizzmo.local',
        port=DEFAULT_COORDINATOR_PORT,
    )

    camera_index: int = 0


config = Config()
