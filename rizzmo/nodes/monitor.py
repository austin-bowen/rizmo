import asyncio
import curses
import time
from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from easymesh import build_mesh_node_from_args
from easymesh.argparse import get_node_arg_parser
from easymesh.asyncio import forever

from rizzmo.config import config
from rizzmo.nodes.image_codec import JpegImageCodec
from rizzmo.nodes.messages import Detection, Detections

Image = np.ndarray

BLOCK_SYMBOLS = '▁▂▃▄▅▆▇█'


@dataclass
class Cache:
    image: Image = None


class Screen:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self._strs = {}

    def addstr(self, row: int, s: str, draw: bool = True) -> None:
        self._strs[row] = s

        if draw:
            self.draw()

    def draw(self) -> None:
        self.stdscr.clear()
        for row, s in self._strs.items():
            self.stdscr.addstr(row, 0, s)
        self.stdscr.refresh()


async def main(args: Namespace, stdscr):
    node = await build_mesh_node_from_args(args=args)

    cache = Cache()

    screen = Screen(stdscr)
    screen.addstr(0, '# Camera')
    screen.addstr(5, '# Audio')

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
        now = time.time()
        latency = now - data.timestamp
        nonlocal fps, t_last
        alpha = 0.1
        fps = alpha * (1 / (now - t_last)) + (1 - alpha) * fps
        t_last = now

        labels = [
            o.label for o in sorted(
                data.objects,
                key=lambda o: o.box.area * o.confidence,
                reverse=True,
            )
        ]

        screen.addstr(1, f'FPS: {fps:.2f}\t Latency: {latency:.2f}', draw=False)
        screen.addstr(2, f'Objects: {labels}')

        image = cache.image
        if image is None:
            return

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

    async def handle_tracking(topic, target: Optional[Detection]):
        screen.addstr(3, f'Tracking: {target.label if target else None}')

    power_history = deque(maxlen=10)

    async def handle_audio(topic, data) -> None:
        audio, timestamp = data

        power = np.abs(audio.data).max()
        power = min(power, 1. - 1e-3)
        power = int(power * len(BLOCK_SYMBOLS))
        power = BLOCK_SYMBOLS[power]

        power_history.append(power)
        power = ''.join(power_history)

        screen.addstr(6, f'Audio power: {power}')

    async def handle_voice_detected(topic, data) -> None:
        audio, timestamp, voice_detected = data
        screen.addstr(7, f'Voice detected: {voice_detected}')

    await node.listen('new_image', handle_new_image)
    await node.listen('objects_detected', handle_obj_detected)
    await node.listen('tracking', handle_tracking)
    await node.listen('audio', handle_audio)
    await node.listen('voice_detected', handle_voice_detected)

    await forever()


def get_args() -> Namespace:
    parser = get_node_arg_parser(
        default_node_name='monitor',
        default_coordinator=config.coordinator,
    )

    return parser.parse_args()


if __name__ == '__main__':
    curses.wrapper(lambda stdscr: asyncio.run(main(get_args(), stdscr)))
