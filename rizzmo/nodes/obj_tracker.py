import asyncio
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
    objects = filter(lambda obj: obj.label in labels, objects)
    return max(objects, default=None, key=area)


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

        print(f'(x, y)_error: {x_error:.2f}, {y_error:.2f}')
        print(f'size: {object_size:.2f}')

        eps = 0.05
        if abs(x_error) <= eps and abs(y_error) <= eps:
            return

        x_error = min(max(-1., 2 * x_error), 1.)

        maestro_cmd = ChangeServoPosition(
            pan_deg=-3 * x_error,
            tilt1_deg=-2 * y_error,
        )

        await maestro_cmd_topic.send(maestro_cmd)

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


if __name__ == '__main__':
    asyncio.run(main())
