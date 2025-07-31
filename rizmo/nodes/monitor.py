import asyncio
import curses
import logging
import time
from argparse import Namespace
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Optional

import cv2
import numpy as np
from rosy import build_node_from_args
from rosy.asyncio import forever

from rizmo.image_codec import JpegImageCodec
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages_py36 import Detection, Detections
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

Image = np.ndarray

BLOCK_SYMBOLS = '▁▂▃▄▅▆▇█'


@dataclass
class Cache:
    image: Image = None
    objects: list[Detection] = field(default_factory=list)


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
    logging.basicConfig(level=args.log)

    node = await build_node_from_args(args=args)

    cache = Cache()

    screen = Screen(stdscr)
    screen.addstr(0, '# Camera')
    screen.addstr(5, '# Audio')

    t_last = time.time()
    fps = 0.

    codec = JpegImageCodec()

    def show_image() -> None:
        image = cache.image
        if image is None:
            return
        image = image.copy()

        for obj in cache.objects:
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

        draw_crosshair(image, width=20, thickness=2, color=(255, 255, 255))
        draw_crosshair(image, width=20, thickness=1, color=(0, 0, 0))

        cv2.imshow(f'Camera & Detected Objects', image)
        cv2.waitKey(1)

    async def handle_new_image(topic, data):
        timestamp, camera_index, image_bytes = data
        cache.image = codec.decode(image_bytes)
        image_ready_event.set()

    async def handle_obj_detected(topic, data: Detections):
        now = time.time()

        cache.objects = data.objects
        image_ready_event.set()

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

    async def handle_tracking(topic, target: Optional[Detection]):
        screen.addstr(3, f'Tracking: {target.label if target else None}')

    power_history = deque(maxlen=10)

    async def handle_audio(topic, data) -> None:
        audio, timestamp = data

        power = np.abs(audio.signal).max()
        power = min(power, 1. - 1e-3)
        power = int(power * len(BLOCK_SYMBOLS))
        power = BLOCK_SYMBOLS[power]

        power_history.append(power)
        power = ''.join(power_history)

        screen.addstr(6, f'Audio power: {power}')

    async def handle_voice_detected(topic, data) -> None:
        audio, timestamp, voice_detected = data
        screen.addstr(7, f'Voice detected: {voice_detected}')

    async def handle_transcript(topic, data) -> None:
        screen.addstr(8, f'Transcript: {data!r}')

    async def handle_say(topic, data: str) -> None:
        screen.addstr(9, f'Said: {data!r}')

    image_ready_event = Event()
    RenderThread(show_image, image_ready_event, fps=30).start()

    await node.listen(Topic.NEW_IMAGE_COMPRESSED, handle_new_image)
    await node.listen(Topic.OBJECTS_DETECTED, handle_obj_detected)
    await node.listen(Topic.TRACKING, handle_tracking)
    await node.listen(Topic.AUDIO, handle_audio)
    await node.listen(Topic.VOICE_DETECTED, handle_voice_detected)
    await node.listen(Topic.TRANSCRIPT, handle_transcript)
    await node.listen(Topic.SAY, handle_say)

    await forever()


def draw_crosshair(image: Image, width: int, thickness: int, color=(128, 128, 128)) -> None:
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    width //= 2

    # Draw horizontal line
    cv2.line(
        image,
        (center_x - width, center_y),
        (center_x + width, center_y),
        color,
        thickness,
    )

    # Draw vertical line
    cv2.line(
        image,
        (center_x, center_y - width),
        (center_x, center_y + width),
        color,
        thickness,
    )


class RenderThread(Thread):
    def __init__(self, render_func, image_ready_event: Event, fps: float):
        super().__init__(daemon=True)
        self.render_func = render_func
        self.image_ready_event = image_ready_event
        self.fps = fps

    def run(self):
        while True:
            t0 = time.monotonic()
            self.image_ready_event.wait()
            self.image_ready_event.clear()

            self.render_func()
            dt = time.monotonic() - t0

            delay = max(0., 1 / self.fps - dt)
            time.sleep(delay)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    _args = parse_args()
    curses.wrapper(lambda stdscr: asyncio.run(main(_args, stdscr)))
