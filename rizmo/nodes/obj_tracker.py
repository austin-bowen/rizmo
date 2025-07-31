import asyncio
import logging
import time
from argparse import Namespace
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from rosy import build_node_from_args
from rosy.asyncio import forever

from rizmo.asyncio import DelayedCallback
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import SetHeadSpeed
from rizmo.nodes.messages_py36 import Detection, Detections
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

AVG_LATENCY = 0.085
"""Gains were tuned with this average latency."""


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    node = await build_node_from_args(args=args)

    tracking_topic = node.get_topic(Topic.TRACKING)
    maestro_cmd_topic = node.get_topic(Topic.MAESTRO_CMD)

    @dataclass
    class State:
        label_priorities: dict[str, int] = field(default_factory=lambda: {
            'cat': 0,
            'dog': 0,
            'face': 1,
            'person': 2,
        })

        max_priority = float('inf')

        last_target: Detection = None

        prev_x_error: float = 0.
        last_t: float = float('-inf')

        dead_zone_growth_time: float = 5.
        max_x_dead_zone: float = 0.2
        max_y_dead_zone: float = 0.2
        x_dead_zone: float = 0.
        y_dead_zone: float = 0.

        async def reset_max_priority(self):
            self.max_priority = float('inf')

    state = State()

    reset_max_priority = DelayedCallback(5, state.reset_max_priority)

    @maestro_cmd_topic.depends_on_listener()
    async def handle_objects_detected(topic, data: Detections):
        now = time.time()
        dt = now - state.last_t
        latency = now - data.timestamp
        image_width, image_height = data.image_size

        target = get_tracked_object(
            data.objects,
            {
                l: p for l, p in state.label_priorities.items()
                if p <= state.max_priority
            },
        )

        print()
        print(f'latency: {latency}')
        print(f'tracking: {target}')

        try:
            if target is None:
                if state.last_target is not None:
                    await tracking_topic.send(None)

                return
        finally:
            state.last_target = target

        state.max_priority = state.label_priorities[target.label]
        await reset_max_priority.reschedule()

        box = target.box

        object_x = box.x + box.width / 2
        # Calculate x_error in same scale as image height
        # so x_error and y_error have same scale
        x_error = (2 * object_x - image_width) / image_height

        object_y = image_height - box.y
        if target.label == 'person':
            object_y -= .3 * box.height if object_y < image_height - 10 else 0
        else:
            object_y -= .5 * box.height
        y_error = (2 * object_y / image_height) - 1

        target_area = 0.15
        object_size = (box.width * box.height) / (image_width * image_height)
        target_dist = target_area ** -0.5
        actual_dist = object_size ** -0.5
        z_error = target_dist - actual_dist

        print(f'(x, y, z)_error: {x_error:.2f}, {y_error:.2f}, {z_error:.2f}')

        if abs(x_error) < state.x_dead_zone:
            x_error = state.prev_x_error = 0
        if abs(x_error) > state.max_x_dead_zone:
            state.x_dead_zone = 0.
        else:
            state.x_dead_zone = min(
                state.x_dead_zone + state.max_x_dead_zone * dt / state.dead_zone_growth_time,
                state.max_x_dead_zone,
            )

        if abs(y_error) < state.y_dead_zone:
            y_error = 0
        if abs(y_error) > state.max_y_dead_zone:
            state.y_dead_zone = 0.
        else:
            state.y_dead_zone = min(
                state.y_dead_zone + state.max_y_dead_zone * dt / state.dead_zone_growth_time,
                state.max_y_dead_zone,
            )

        # PD control
        pan_dps = 50 * x_error + 10 * (x_error - state.prev_x_error) / dt
        tilt_dps = 50 * y_error
        lean_dps = 25 * z_error

        # This decreases gain as latency increases to prevent overshooting
        gain_scalar = AVG_LATENCY / latency
        state.prev_x_error = x_error
        state.last_t = now

        maestro_cmd = SetHeadSpeed(
            pan_dps=-pan_dps * gain_scalar,
            tilt_dps=-tilt_dps * gain_scalar,
            lean_dps=lean_dps * gain_scalar,
        )

        await maestro_cmd_topic.send(maestro_cmd)
        await tracking_topic.send(target)

    await node.listen(Topic.OBJECTS_DETECTED, handle_objects_detected)

    await forever()


def get_tracked_object(
        objects: Iterable[Detection],
        label_priorities: dict[str, int],
) -> Optional[Detection]:
    objects = filter(lambda obj: obj.label in label_priorities.keys(), objects)

    return max(
        objects,
        key=lambda o: (
            -label_priorities[o.label],
            o.box.area,
        ),
        default=None,
    )


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
