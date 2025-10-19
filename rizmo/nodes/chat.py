import asyncio
import logging
import sys
from argparse import Namespace

from rosy import Node, build_node_from_args

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    async with await build_node_from_args(args=args) as node:
        await _main(args, node)


async def _main(args: Namespace, node: Node) -> None:
    print("Waiting for listener...")
    transcript_topic = node.get_topic(Topic.TRANSCRIPT)
    await transcript_topic.wait_for_listener()

    async def handle_say(topic, message: str) -> None:
        print(f"[Rizmo] {message}")
        print("\n> ", end="", flush=True)

    await node.listen(Topic.SAY, handle_say)

    print("> ", end="", flush=True)
    while True:
        line = await asyncio.to_thread(sys.stdin.readline)
        line = line.strip()
        if line:
            await transcript_topic.send(line)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == "__main__":
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
