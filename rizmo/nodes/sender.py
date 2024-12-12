import asyncio
from argparse import Namespace

from easymesh import build_mesh_node_from_args

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    print(f'Sending message on {args.topic!r}: {args.message!r}')
    await node.wait_for_listener(args.topic)
    await node.send(args.topic, args.message)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        'topic',
        help='The topic to send to.',
    )

    parser.add_argument(
        'message',
        help='The message to send.',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
