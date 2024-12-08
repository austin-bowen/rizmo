from argparse import ArgumentParser
from pathlib import Path

from easymesh.argparse import get_node_arg_parser

from rizmo.config import config


def get_rizmo_node_arg_parser(node_file: str) -> ArgumentParser:
    name = Path(node_file).name.removesuffix('.py')

    return get_node_arg_parser(
        default_node_name=name,
        default_coordinator=config.mesh_coordinator,
        default_authkey=config.mesh_authkey,
    )
