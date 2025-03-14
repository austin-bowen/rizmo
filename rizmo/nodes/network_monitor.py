import asyncio
from argparse import Namespace

from easymesh import build_mesh_node_from_args

from rizmo.network_manager import NetworkManager
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm
from rizmo.config import config


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    network_manager = NetworkManager()

    network_connected_topic = node.get_topic_sender(Topic.NETWORK_CONNECTED)
    say_topic = node.get_topic_sender(Topic.SAY)

    prev_connected = True
    while True:
        connected = await network_manager.connection_is_active(args.ssid)
        await network_connected_topic.send(connected)

        if not connected:
            if prev_connected:
                await say_topic.send('Network disconnected; attempting to reconnect.')

            print('Network disconnected; attempting to reconnect...')
            try:
                await network_manager.connect(args.ssid)
            except RuntimeError as e:
                print(repr(e))
            else:
                print('Success!')

        await asyncio.sleep(args.period)

        prev_connected = connected


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--ssid',
        default=config.wifi_ssid,
        help='The SSID of the wifi network to connect to. Default: %(default)s',
    )

    parser.add_argument(
        '--period',
        default=10,
        type=int,
        help='Seconds to wait between connection checks. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
