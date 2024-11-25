import asyncio
from collections.abc import Iterable
from typing import Optional

from easymesh import build_mesh_node
from easymesh.asyncio import forever

from rizzmo.config import config
from rizzmo.nodes.maestro_ctl import ChangeServoPosition
from rizzmo.nodes.messages import Detection


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
        print(f'x_error: {x_error}')

        if abs(x_error) <= 0.1:
            return

        maestro_cmd = ChangeServoPosition(
            pan_deg=-1 if x_error > 0 else 1,
        )

        await maestro_cmd_topic.send(maestro_cmd)

    await node.listen('objects_detected', handle_objects_detected)

    await forever()


if __name__ == '__main__':
    asyncio.run(main())
