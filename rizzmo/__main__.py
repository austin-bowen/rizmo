from time import sleep

from rizzmo.procman import ProcessManager


def main():
    with ProcessManager() as p:
        coordinator = p.start_python_module('easymesh.coordinator')
        sleep(1)

        for node in (
                'mic_node',
                'audio_viz_node',
        ):
            p.start_python_module(f'rizzmo.nodes.{node}')

        try:
            coordinator.wait()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
