import asyncio
import time
from argparse import Namespace
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever

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
        prev_x_error: float = 0.
        last_t: float = float('-inf')

    cache = Cache()

    @maestro_cmd_topic.depends_on_listener()
    async def handle_objects_detected(topic, data: Detections):
        now = time.time()
        latency = now - data.timestamp
        image_width, image_height = data.image_size

        target = get_tracked_object(data.objects)
        await tracking_topic.send(target)

        print()
        print(f'latency: {latency}')
        print(f'tracking: {target}')

        if target is None:
            return

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

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


def get_tracked_object(
        objects: Iterable[Detection],
        labels=(
                'cat',
                'dog',
                'person',
        ),
) -> Optional[Detection]:
    labels_set = set(labels)
    objects = filter(lambda obj: obj.label in labels_set, objects)

    label_priorities = {l: len(labels) - i for i, l in enumerate(labels)}

    return max(
        objects,
        key=lambda o: (
            label_priorities[o.label],
            o.box.area,
        ),
        default=None,
    )


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('obj-tracker')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
