import argparse
import os
import socket
from argparse import Namespace
from time import sleep

from rizmo.config import IS_RIZMO, config
from rizmo.procman import ProcessManager

host_nodes = {
    'rizmo': (
        'camera',
        'cmd_proc',
        'maestro_ctl',
        'mic',
        'obj_tracker',
        'speech',
        'website',
    ),
    'potato': (
        ('obj_rec', 2),
        'vad',
        'asr',
    ),
    'austin-laptop': (
        'monitor',
    )
}


def main(args: Namespace):
    host = socket.gethostname()
    nodes_to_start = host_nodes[host]
    print(f'Starting nodes: {nodes_to_start}')

    os.makedirs('logs', exist_ok=True)

    with ProcessManager() as p:
        if IS_RIZMO:
            p.start_python(
                '-m', 'easymesh.coordinator',
                '--authkey', config.mesh_authkey,
            )
            sleep(1)

        for node in nodes_to_start:
            if isinstance(node, str):
                count = 1
            else:
                node, count = node

            if node in args.exclude:
                print(f'Skipping {node}')
                continue

            # Have to use shell mode to redirect output to a file
            with p.options(shell=True):
                for _ in range(count):
                    p.start_python(
                        '-u', '-m', f'rizmo.nodes.{node}',
                        '|', 'tee', f'logs/{node}.log',
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
