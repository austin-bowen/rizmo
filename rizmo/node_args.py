from argparse import ArgumentParser
from pathlib import Path

from rosy.argparse import get_node_arg_parser
from rosy.cli.utils import add_log_arg

from rizmo.config import config


def get_rizmo_node_arg_parser(node_file: str) -> ArgumentParser:
    node_file = Path(node_file)
    name = node_file.name

    # ".../<node>/__main__.py" -> "<node>"
    if name == '__main__.py':
        name = node_file.parent.name
    # ".../<node>.py" -> "<node>"
    else:
        name = name.removesuffix('.py')

    parser = get_node_arg_parser(
        default_node_name=name,
        default_coordinator=config.mesh_coordinator,
        default_authkey=config.mesh_authkey,
    )

    add_log_arg(parser)

    return parser
