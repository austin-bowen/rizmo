from argparse import ArgumentParser

from easymesh.argparse import get_node_arg_parser

from rizmo.config import config


def get_rizmo_node_arg_parser(name: str) -> ArgumentParser:
    return get_node_arg_parser(
        default_node_name=name,
        default_coordinator=config.mesh_coordinator,
        default_authkey=config.mesh_authkey,
    )
