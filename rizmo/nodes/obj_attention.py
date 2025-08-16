import asyncio
import logging
import random
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

from rosy import build_node_from_args

from rizmo.asyncio import DelayedCallback
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import SetServoPosition
from rizmo.nodes.messages_py36 import Detection
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    async with await build_node_from_args(args=args) as node:
        await _main(node)


async def _main(node) -> None:
    maestro_cmd_topic = node.get_topic(Topic.MAESTRO_CMD)

    @dataclass
    class State:
        target: Detection = None
        camera_is_covered: bool = False

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

    explore = DelayedCallback(5, _explore)
    await explore.schedule()

    async def handle_tracking(topic, target: Optional[Detection]) -> None:
        new_label = target.label if target else None
        old_label = state.target.label if state.target else None
        state.target = target
        if new_label != old_label:
            print(f'Target: {new_label}')

        await explore.set(target is None and not state.camera_is_covered)

    async def handle_camera_covered(topic, covered: bool) -> None:
        state.camera_is_covered = covered
        print(f'Camera covered: {covered}')

        await explore.set(not covered)

        if covered:
            await maestro_cmd_topic.send(SetServoPosition(
                pan_deg=0,
                tilt0_deg=0,
                tilt1_deg=0,
                speed_dps=30,
            ))

    await node.listen(Topic.TRACKING, handle_tracking)
    await node.listen(Topic.CAMERA_COVERED, handle_camera_covered)
    await node.forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
