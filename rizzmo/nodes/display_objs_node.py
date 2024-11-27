import asyncio
import time
from dataclasses import dataclass

import cv2
import numpy as np
from easymesh import build_mesh_node
from easymesh.asyncio import forever

from rizzmo.config import config
from rizzmo.nodes.image_codec import JpegImageCodec
from rizzmo.nodes.messages import Detections

Image = np.ndarray


@dataclass
class Cache:
    image: Image = None


async def main():
    node = await build_mesh_node(
        name='display_objs',
        coordinator_host=config.coordinator_host,
    )

    cache = Cache()

    t_last = time.time()
    fps = 0.

    codec = JpegImageCodec()

    def show_image() -> None:
        cv2.imshow(f'Camera & Detected Objects', cache.image)
        cv2.waitKey(1)

    async def handle_new_image(topic, data):
        timestamp, camera_index, image_bytes = data
        cache.image = codec.decode(image_bytes)

    async def handle_obj_detected(topic, data: Detections):
        image = cache.image
        if image is None:
            return

        now = time.time()
        latency = now - data.timestamp
        nonlocal fps, t_last
        alpha = 0.1
        fps = alpha * (1 / (now - t_last)) + (1 - alpha) * fps
        t_last = now
        print()
        print(f'FPS    : {fps:.2f}')
        print(f'Latency: {latency}')
        print(f'Objects: {data.objects}')

        for obj in data.objects:
            box = obj.box

            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 0, 255),
                2,
            )

            text_y = box.y - 10 if box.y - 10 > 10 else box.y + 20
            cv2.putText(
                image,
                f'{obj.label} {obj.confidence:.2f}',
                (box.x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                image,
                f'{obj.label} {obj.confidence:.2f}',
                (box.x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                1,
            )

        show_image()

    await node.listen('new_image', handle_new_image)
    await node.listen('objects_detected', handle_obj_detected)

    await forever()


if __name__ == '__main__':
    asyncio.run(main())
