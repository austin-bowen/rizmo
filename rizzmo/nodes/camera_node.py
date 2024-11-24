import asyncio
import time
from typing import Any, Optional

import cv2
import numpy as np

from easymesh import build_mesh_node
from easymesh.utils import require
from rizzmo.nodes.image_codec import JpegImageCodec

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


async def main():
    node = await build_mesh_node(
        name='camera',
        # coordinator_host='potato.local',
        # load_balancer=RandomLoadBalancer(),
    )

    await asyncio.gather(*(
        _read_camera(node, camera_index) for camera_index in [
        0,
        # 4,
    ]
    ))


async def _read_camera(
        node,
        camera_index: int,
        camera_fps: float = 30,
        fps_limit: Optional[float] = 10,
        show_raw_image: bool = False,
):
    new_image_topic = node.get_topic_sender('new_image')
    await new_image_topic.wait_for_listener()

    camera_builder = lambda: Camera(
        camera_index,
        # resolution=(640, 360),
        resolution=(1280, 720),
        # resolution=(1920, 1080),
        fps=camera_fps,
        codec='MJPG',
        props={
            cv2.CAP_PROP_AUTO_WB: 0,
            cv2.CAP_PROP_WB_TEMPERATURE: 2800,
        }
    )
    camera = camera_builder()

    codec = JpegImageCodec(quality=80)

    while True:
        t0 = time.monotonic()

        try:
            image = await camera.get_image()
        except CameraCaptureError as e:
            print(f'Error reading camera {camera_index}: {e}')
            await asyncio.sleep(1)
            continue

        timestamp = time.time()
        image = image[::2, ::2]
        image_bytes = codec.encode(image)
        await new_image_topic.send((timestamp, camera_index, image_bytes))

        print('.', end='', flush=True)

        if show_raw_image:
            cv2.imshow(f'Camera {camera_index}: Raw Image', image)
            cv2.waitKey(1)

        if not await new_image_topic.has_listeners():
            await camera.close()
            await new_image_topic.wait_for_listener(.1)
            camera = camera_builder()

        if fps_limit is not None:
            wait_time = 1 / fps_limit - (time.monotonic() - t0)
            if wait_time > 0:
                await asyncio.sleep(wait_time)


if __name__ == '__main__':
    asyncio.run(main())