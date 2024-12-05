import asyncio
import time
from argparse import Namespace
from typing import Any, Optional

import cv2
import numpy as np
from easymesh import build_mesh_node_from_args
from easymesh.utils import require

from rizmo.asyncio import DelayedCallback
from rizmo.config import config
from rizmo.motion_detector import DynamicThresholdPixelChangeMotionDetector
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.image_codec import JpegImageCodec
from rizmo.signal import graceful_shutdown_on_sigterm

Image = np.ndarray


class VideoCapture(cv2.VideoCapture):
    def set(self, propId: int, value, verify: bool = False) -> bool:
        success = super().set(propId, value)

        if verify:
            actual = self.get(propId)
            require(
                success and actual == value,
                f'Failed to set property {propId} to {value!r}; '
                f'current value is {actual!r}'
            )

        return success


class Camera:
    def __init__(
            self,
            device: int,
            resolution: tuple[int, int] = None,
            fps: float = None,
            codec: str = None,
            props: dict[int, Any] = None,
    ):
        self.device = device
        self.resolution = resolution
        self.fps = fps
        self.codec = codec
        self.props = props or {}

        self._capture = None

    async def get_image(self) -> Image:
        capture = self._get_capture()

        ret, frame = await asyncio.to_thread(capture.read)

        if not ret:
            await self.close()
            raise CameraCaptureError('Failed to capture image')

        return frame

    def _get_capture(self) -> cv2.VideoCapture:
        if self._capture is not None:
            return self._capture

        self._capture = VideoCapture(self.device)

        if not self._capture.isOpened():
            raise CameraCaptureError(f'Failed to open camera device {self.device}')

        if self.codec is not None:
            # noinspection PyUnresolvedReferences
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._capture.set(cv2.CAP_PROP_FOURCC, fourcc, verify=True)

        if self.resolution is not None:
            width, height = self.resolution
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width, verify=False)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height, verify=True)

        if self.fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self.fps, verify=True)

        for prop_id, value in self.props.items():
            self._capture.set(prop_id, value, verify=True)

        return self._capture

    async def close(self) -> None:
        if self._capture is not None:
            capture = self._capture
            self._capture = None

            await asyncio.to_thread(capture.release)


class CameraCaptureError(Exception):
    pass


async def main(args: Namespace) -> None:
    node = await build_mesh_node_from_args(args=args)

    await _read_camera(
        node,
        args.camera_index,
        args.resolution,
        args.camera_fps,
        fps_limit=args.fps_limit,
        show_raw_image=args.show_raw_image,
        jpeg_quality=args.jpeg_quality,
    )


async def _read_camera(
        node,
        camera_index: int,
        resolution: tuple[int, int],
        camera_fps: float,
        fps_limit: Optional[float] = None,
        show_raw_image: bool = False,
        jpeg_quality: int = 80,
):
    new_image_topic = node.get_topic_sender('new_image')
    await new_image_topic.wait_for_listener()

    camera_builder = lambda: Camera(
        camera_index,
        resolution,
        fps=camera_fps,
        codec='MJPG',
        props={
            cv2.CAP_PROP_AUTO_WB: 0,
            cv2.CAP_PROP_WB_TEMPERATURE: 2800,
        }
    )
    camera = camera_builder()

    motion_detector = DynamicThresholdPixelChangeMotionDetector(
        change=0.1,
        alpha=0.01,
        subsample=8,
    )

    codec = JpegImageCodec(quality=jpeg_quality)

    min_fps, max_fps = 5, fps_limit

    class Cache:
        fps_limit: float = max_fps
        prev_motion: bool = False

    cache = Cache()

    async def low_fps():
        print(f'\nSwitching to low FPS: {min_fps}')
        cache.fps_limit = min_fps

    delayed_low_fps = DelayedCallback(3, low_fps)

    while True:
        t0 = time.monotonic()

        try:
            image = await camera.get_image()
        except CameraCaptureError as e:
            print(f'Error reading camera {camera_index}: {e}')
            await asyncio.sleep(1)
            continue

        timestamp = time.time()
        image_bytes = codec.encode(image)
        await new_image_topic.send((timestamp, camera_index, image_bytes))

        print('.', end='', flush=True)

        motion = motion_detector.is_motion(image)
        if motion != cache.prev_motion:
            cache.prev_motion = motion
            print(f'\nMotion: {motion}')

        if motion:
            await delayed_low_fps.cancel()
            print(f'\nSwitching to high FPS: {max_fps}')
            cache.fps_limit = max_fps
        else:
            await delayed_low_fps.schedule()

        if show_raw_image:
            cv2.imshow(f'Camera {camera_index}: Raw Image', image)
            cv2.waitKey(1)

        if not await new_image_topic.has_listeners():
            await camera.close()
            await new_image_topic.wait_for_listener(.1)
            camera = camera_builder()

        if cache.fps_limit is not None:
            wait_time = 1 / cache.fps_limit - (time.monotonic() - t0)
            if wait_time > 0:
                await asyncio.sleep(wait_time)


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('camera')

    parser.add_argument(
        '--camera-index', '-c',
        default=config.camera_index,
        type=int,
        help='Camera index. Default: %(default)s',
    )

    def resolution_type(value: str) -> tuple[int, int]:
        w, h = value.split(',', maxsplit=1)
        return int(w), int(h)

    parser.add_argument(
        '--resolution', '-r',
        default=config.camera_resolution,
        type=resolution_type,
        metavar='WIDTH,HEIGHT',
        help='Camera resolution. Default: %(default)s. '
             'Other options: 640,360; 1920,1080',
    )

    parser.add_argument(
        '--camera-fps', '-f',
        default=30.,
        type=float,
        help='Camera FPS. Default: %(default)s',
    )

    parser.add_argument(
        '--fps-limit', '-l',
        type=float,
        help='FPS limit',
    )

    parser.add_argument(
        '--show-raw-image',
        action='store_true',
    )

    parser.add_argument(
        '--jpeg-quality',
        default=80,
        type=int,
        help='JPEG quality. Default: %(default)s. Range: 0-100',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
