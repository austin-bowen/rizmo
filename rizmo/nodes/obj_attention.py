import asyncio
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

from easymesh import build_mesh_node_from_args

from rizmo.asyncio import DelayedCallback
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import SetServoPosition
from rizmo.nodes.topics import Topic
from rizmo.nodes.messages_py36 import Detection
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    maestro_cmd_topic = node.get_topic_sender(Topic.MAESTRO_CMD)

    @dataclass
    class State:
        target: Detection = None

    state = State()

    async def _explore():
        delay = 1.
        while True:
            print('Exploring...')
            await maestro_cmd_topic.send(SetServoPosition(
                pan_deg=random.uniform(-120, 120),
                tilt0_deg=0,
                tilt1_deg=random.uniform(0, 45),
                speed_dps=15,
            ))

            await asyncio.sleep(delay)
            delay *= 2
            delay = min(delay, 60)

    explore = DelayedCallback(1, _explore)
    await explore.schedule()

    async def handle_tracking(topic, target: Optional[Detection]) -> None:
        state.target = target
        await explore.set(target is None)

    await node.listen(Topic.TRACKING, handle_tracking)

    while True:
        print('target:', state.target)
        await asyncio.sleep(1)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
