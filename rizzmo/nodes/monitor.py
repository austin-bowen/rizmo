import asyncio
import asyncio
import time
from argparse import Namespace
from dataclasses import dataclass

import cv2
import numpy as np
from easymesh import build_mesh_node_from_args
from easymesh.argparse import get_node_arg_parser
from easymesh.asyncio import forever

from rizzmo.config import config
from rizzmo.nodes.image_codec import JpegImageCodec
from rizzmo.nodes.messages import Detections

Image = np.ndarray

BLOCK_SYMBOLS = '▁▂▃▄▅▆▇█'
TOPIC_OPTIONS = {'new_image', 'objects_detected', 'audio'}


@dataclass
class Cache:
    image: Image = None


async def main(args: Namespace):
    node = await build_mesh_node_from_args(args=args)

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

    async def handle_audio(topic, data) -> None:
        audio, timestamp = data

        power = np.abs(audio.data).max()
        power = min(power, 1. - 1e-3)
        power = int(power * len(BLOCK_SYMBOLS))
        power = BLOCK_SYMBOLS[power]
        print(power, end='', flush=True)

    topics_and_handlers = {
        'new_image': handle_new_image,
        'objects_detected': handle_obj_detected,
        'audio': handle_audio,
    }
    assert set(topics_and_handlers.keys()) == TOPIC_OPTIONS

    ignore_topics = args.ignore_topics or set()

    for topic, handler in topics_and_handlers.items():
        if topic not in ignore_topics:
            await node.listen(topic, handler)

    await forever()


def get_args() -> Namespace:
    parser = get_node_arg_parser(
        default_node_name='monitor',
        default_coordinator=config.coordinator,
    )

    parser.add_argument(
        '--ignore-topics', '-i',
        nargs='+',
        choices=TOPIC_OPTIONS,
        help='The topics to ignore. Default: None',
    )

    return parser.parse_args()


if __name__ == '__main__':
    asyncio.run(main(get_args()))
