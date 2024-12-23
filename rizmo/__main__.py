import argparse
import os
import socket
from argparse import Namespace
from dataclasses import dataclass, field
from time import sleep

from rizmo.config import IS_RIZMO, config
from rizmo.procman import ProcessManager


@dataclass
class Node:
    name: str
    args: list[str] = field(default_factory=list)
    count: int = 1


host_nodes = {
    'rizmo': (
        Node('agent'),
        Node('camera'),
        Node('maestro_ctl'),
        Node('mic'),
        Node('obj_tracker'),
        Node('voice'),
        Node('website'),
    ),
    'potato': (
        Node('obj_rec'),
        Node('vad'),
        Node('asr'),
    ),
    'austin-laptop': (
        Node('monitor'),
    )
}


def main(args: Namespace):
    host = socket.gethostname()
    nodes_to_start = host_nodes[host]
    print(f'Starting nodes: {[n.name for n in nodes_to_start]}')

    os.makedirs('logs', exist_ok=True)

    with ProcessManager() as p:
        if IS_RIZMO:
            p.start_python(
                '-m', 'easymesh.coordinator',
                '--authkey', config.mesh_authkey,
            )

            p.popen('./bin/start-py36-server')

            sleep(1)

        for node in nodes_to_start:
            if node.name in args.exclude:
                print(f'Skipping {node.name}')
                continue

            # Have to use shell mode to redirect output to a file
            with p.options(shell=True):
                for _ in range(node.count):
                    p.start_python(
                        '-u', '-m', f'rizmo.nodes.{node.name}',
                        *node.args,
                        '|', 'tee', f'logs/{node.name}.log',
                    )

        try:
            p.wait()
        except KeyboardInterrupt:
            pass


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help='Nodes to exclude from starting',
    )

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
