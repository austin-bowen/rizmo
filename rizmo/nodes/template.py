import asyncio
import logging
from argparse import Namespace

from rosy import Node, build_node_from_args

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    async with await build_node_from_args(args=args) as node:
        await _main(args, node)


async def _main(args: Namespace, node: Node) -> None:
    ...


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
