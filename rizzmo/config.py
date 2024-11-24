from dataclasses import dataclass

from easymesh.types import Host


@dataclass
class Config:
    coordinator_host: Host = 'rizzmo.local'


config = Config()
