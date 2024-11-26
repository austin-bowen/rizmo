import asyncio
import time
from collections.abc import Iterable
from typing import Optional

from easymesh import build_mesh_node
from easymesh.asyncio import forever

from rizzmo.config import config
from rizzmo.nodes.messages import ChangeServoPosition, Detection


def area(detection: Detection) -> float:
    return detection.box.width * detection.box.height


def get_tracked_object(
        objects: Iterable[Detection],
        labels=(
                'cat',
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
            area(o),
        ),
        default=None,
    )


async def main():
    node = await build_mesh_node(
        name='obj-tracker',
        coordinator_host=config.coordinator_host,
    )

    maestro_cmd_topic = node.get_topic_sender('maestro_cmd')

    @maestro_cmd_topic.depends_on_listener()
    async def handle_objects_detected(topic, data):
        timestamp, camera_index, image_bytes, objects = data
        image_width, image_height = 1280 / 2, 720 / 2

        tracked_object = get_tracked_object(objects)

        print()
        print(f'latency: {timestamp - time.time()}')
        print(f'tracking: {tracked_object}')
        if tracked_object is None:
            return

        box = tracked_object.box

        object_x = box.x + box.width / 2
        x_error = (2 * object_x / image_width) - 1

        object_y = image_height - box.y
        if tracked_object.label == 'person':
            object_y -= .3 * box.height
        else:
            object_y -= .5 * box.height
        y_error = (2 * object_y / image_height) - 1

        object_size = (box.width * box.height) / (image_width * image_height)
        z_error = min(max(-1., 3 * object_size - 1), 1.)

        print(f'(x, y, z)_error: {x_error:.2f}, {y_error:.2f}, {z_error:.2f}')

        if (x_error ** 2 + y_error ** 2) ** 0.5 <= 0.1:
            x_error = y_error = 0

        x_error = min(max(-1., 1.5 * x_error), 1.)

        maestro_cmd = ChangeServoPosition(
            pan_deg=-3 * x_error,
            tilt0_deg=1 * z_error,
            tilt1_deg=-2 * y_error,
        )

        await maestro_cmd_topic.send(maestro_cmd)

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


if __name__ == '__main__':
    asyncio.run(main())
