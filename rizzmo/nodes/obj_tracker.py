import asyncio
from collections.abc import Iterable
from typing import Optional

from easymesh import build_mesh_node
from easymesh.asyncio import forever

from rizzmo.config import config
from rizzmo.nodes.messages import Box, Detection


def area(box: Box) -> float:
    return box.width * box.height


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

    todo_topic = node.get_topic_sender('TODO')

    # @todo_topic.depends_on_listener()
    async def handle_objects_detected(topic, data):
        timestamp, camera_index, image_bytes, objects = data
        image_width, image_height = 1280 / 2, 720 / 2

        tracked_object: Detection = get_tracked_object(objects)
        print(f'tracking: {tracked_object}')

        object_x = tracked_object.box.x + tracked_object.box.width / 2
        x_error = (2 * object_x / image_width) - 1
        print(f'x_error: {x_error}')

        await todo_topic.send(...)

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


if __name__ == '__main__':
    asyncio.run(main())
