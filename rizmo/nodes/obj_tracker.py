import asyncio
import time
from argparse import Namespace
from collections.abc import Iterable
from typing import Optional

from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import ChangeServoPosition, Detection, Detections
from rizmo.signal import graceful_shutdown_on_sigterm

AVG_LATENCY = 0.0463
"""Gains were tuned with this average latency."""


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


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    tracking_topic = node.get_topic_sender('tracking')
    maestro_cmd_topic = node.get_topic_sender('maestro_cmd')

    high_fps, low_fps = 30, 5
    low_fps_future = None

    async def set_fps(fps: float) -> None:
        await node.send('set_fps_limit', fps)

    async def go_low_fps() -> None:
        await asyncio.sleep(3)
        await set_fps(low_fps)

    async def go_high_fps() -> None:
        await set_fps(high_fps)

    @maestro_cmd_topic.depends_on_listener()
    async def handle_objects_detected(topic, data: Detections):
        nonlocal low_fps_future

        latency = time.time() - data.timestamp
        image_width, image_height = 1280 / 2, 720 / 2

        target = get_tracked_object(data.objects)
        await tracking_topic.send(target)

        print()
        print(f'latency: {latency}')
        print(f'tracking: {target}')
        if target is None:
            if low_fps_future is None:
                low_fps_future = asyncio.create_task(go_low_fps())

            return

        box = target.box

        object_x = box.x + box.width / 2
        # Calculate x_error in same scale as image height
        # so x_error and y_error have same scale
        x_error = (2 * object_x - image_width) / image_height

        object_y = image_height - box.y
        if target.label == 'person':
            object_y -= .3 * box.height
        else:
            object_y -= .5 * box.height
        y_error = (2 * object_y / image_height) - 1

        object_size = (box.width * box.height) / (image_width * image_height)
        z_error = min(max(-1., 3 * object_size - 1), 1.)

        print(f'(x, y, z)_error: {x_error:.2f}, {y_error:.2f}, {z_error:.2f}')

        if (x_error ** 2 + y_error ** 2) ** 0.5 <= 0.1:
            x_error = y_error = 0

            if low_fps_future is None:
                low_fps_future = asyncio.create_task(go_low_fps())
        else:
            if low_fps_future is not None:
                low_fps_future.cancel()
                low_fps_future = None
                await go_high_fps()

        # This decreases gain as latency increases to prevent overshooting
        gain_scalar = AVG_LATENCY / latency
        x_gain = -2 * gain_scalar
        y_gain = -2 * gain_scalar
        z_gain = 1 * gain_scalar

        maestro_cmd = ChangeServoPosition(
            pan_deg=x_gain * x_error,
            tilt0_deg=z_gain * z_error,
            tilt1_deg=y_gain * y_error,
        )

        await maestro_cmd_topic.send(maestro_cmd)

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('obj-tracker')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
