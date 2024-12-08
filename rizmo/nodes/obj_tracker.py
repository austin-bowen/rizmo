import asyncio
import time
from argparse import Namespace
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever

from rizmo.asyncio import DelayedCallback
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import Detection, Detections, SetHeadSpeed
from rizmo.signal import graceful_shutdown_on_sigterm

AVG_LATENCY = 0.0367
"""Gains were tuned with this average latency."""


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    tracking_topic = node.get_topic_sender('tracking')
    maestro_cmd_topic = node.get_topic_sender('maestro_cmd')

    @dataclass
    class Cache:
        label_priorities: dict[str, int] = field(default_factory=lambda: {
            'cat': 0,
            'dog': 0,
            'person': 1,
        })
        max_priority = float('inf')

        last_target: Detection = None

        prev_x_error: float = 0.
        last_t: float = float('-inf')

        async def reset_max_priority(self):
            self.max_priority = float('inf')

    cache = Cache()

    reset_max_priority = DelayedCallback(5, cache.reset_max_priority)

    @maestro_cmd_topic.depends_on_listener()
    async def handle_objects_detected(topic, data: Detections):
        now = time.time()
        latency = now - data.timestamp
        image_width, image_height = data.image_size

        target = get_tracked_object(
            data.objects,
            {
                l: p for l, p in cache.label_priorities.items()
                if p <= cache.max_priority
            },
        )

        print()
        print(f'latency: {latency}')
        print(f'tracking: {target}')

        try:
            if target is None:
                if cache.last_target is not None:
                    await tracking_topic.send(None)

                return
        finally:
            cache.last_target = target

        cache.max_priority = cache.label_priorities[target.label]
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

        object_size = (box.width * box.height) / (image_width * image_height)
        z_error = min(max(-1., 3 * object_size - 1), 1.)

        print(f'(x, y, z)_error: {x_error:.2f}, {y_error:.2f}, {z_error:.2f}')

        if abs(x_error) < 0.15:
            x_error = cache.prev_x_error = 0
        if abs(y_error) < 0.15:
            y_error = 0

        # PD control
        dt = now - cache.last_t
        pan_dps = 150 * x_error + 20 * (x_error - cache.prev_x_error) / dt
        tilt_dps = 90 * y_error

        # This decreases gain as latency increases to prevent overshooting
        gain_scalar = AVG_LATENCY / latency
        cache.prev_x_error = x_error
        cache.last_t = now

        maestro_cmd = SetHeadSpeed(
            pan_dps=-pan_dps * gain_scalar,
            tilt_dps=-tilt_dps * gain_scalar,
        )

        await maestro_cmd_topic.send(maestro_cmd)
        await tracking_topic.send(target)

    await node.listen('objects_detected', handle_objects_detected)

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
