import asyncio
from argparse import Namespace

from easymesh import build_node_from_args

from rizmo.config import config
from rizmo.network_manager import NetworkManager
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await build_node_from_args(args=args)

    network_manager = NetworkManager()

    network_connected_topic = node.get_topic(Topic.NETWORK_CONNECTED)
    say_topic = node.get_topic(Topic.SAY)

    async def say(message: str) -> None:
        print(message)
        await say_topic.send(message)

    async def wifi_is_connected() -> bool:
        return await network_manager.device_is_connected(args.wifi_device)

    async def wait_wifi_connected() -> None:
        while not await wifi_is_connected():
            await asyncio.sleep(5)

    prev_connected = True
    while True:
        connected = await wifi_is_connected()
        await network_connected_topic.send(connected)

        if connected and not prev_connected:
            await say('Network reconnected!')

        if not connected:
            await say('Network disconnected; attempting to reconnect...')
            try:
                await network_manager.cycle_wifi_radio_powered()
            except RuntimeError as e:
                print(repr(e))

            try:
                await asyncio.wait_for(wait_wifi_connected(), timeout=30)
            except asyncio.TimeoutError:
                pass

        await asyncio.sleep(10)

        prev_connected = connected


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--wifi-device',
        default=config.wifi_device,
        help='The WiFi device name. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
