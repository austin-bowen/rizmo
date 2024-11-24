import socket
from time import sleep

from rizzmo.config import IS_RIZZMO
from rizzmo.procman import ProcessManager

host_nodes = {
    'rizzmo': (
        'camera_node',
        'mic_node',
        'website_node',
    ),
    'potato': (
        'objrec_node',
        'vad_node',
    ),
    'austin-laptop': (
        'display_objs_node',
        'audio_viz_node',
    )
}


def main():
    host = socket.gethostname()
    nodes_to_start = host_nodes[host]

    with ProcessManager() as p:
        if IS_RIZZMO:
            p.start_python_module('easymesh.coordinator')
            sleep(1)

        for node in nodes_to_start:
            p.start_python_module(f'rizzmo.nodes.{node}')

        try:
            p.wait()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
