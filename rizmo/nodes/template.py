import asyncio
from argparse import Namespace

from easymesh import build_mesh_node_from_args

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
