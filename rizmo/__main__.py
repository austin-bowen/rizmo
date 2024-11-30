import argparse
import socket
from argparse import Namespace
from time import sleep

from rizmo.config import IS_RIZMO
from rizmo.procman import ProcessManager

host_nodes = {
    'rizmo': (
        'camera_node',
        'cmd_proc',
        'maestro_ctl',
        'mic_node',
        'obj_tracker',
        'speech_node',
        'website_node',
    ),
    'potato': (
        ('objrec_node', 1),
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

    with ProcessManager() as p:
        if IS_RIZMO:
            p.start_python_module('easymesh.coordinator')
            sleep(1)

        for node in nodes_to_start:
            if isinstance(node, str):
                count = 1
            else:
                node, count = node

            if node in args.exclude:
                print(f'Skipping {node}')
                continue

            for _ in range(count):
                p.start_python_module(f'rizmo.nodes.{node}')

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
