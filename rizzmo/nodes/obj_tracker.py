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
        print(f'tracking: {tracked_object}')
        if tracked_object is None:
            return

        object_x = tracked_object.box.x + tracked_object.box.width / 2
        x_error = (2 * object_x / image_width) - 1

        if tracked_object.label == 'person':
            object_y = tracked_object.box.y# - .2 * tracked_object.box.height
        else:
            object_y = tracked_object.box.y# + tracked_object.box.height / 2
        y_error = object_y

        print(f'(x, y)_error: {x_error:.2f}, {y_error:.2f}')

        if abs(x_error) <= 0.05:
            return

        x_error = min(max(-1., 2 * x_error), 1.)
        delta = 3.
        maestro_cmd = ChangeServoPosition(
            pan_deg=-delta * x_error,
        )

        await maestro_cmd_topic.send(maestro_cmd)

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


if __name__ == '__main__':
    asyncio.run(main())
