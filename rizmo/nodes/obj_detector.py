import asyncio
import logging
from abc import abstractmethod
from argparse import Namespace
from typing import Optional, Union

import cv2
import numpy as np
import torch
import ultralytics
from PIL import Image as PILImage
from rosy import build_node_from_args

from rizmo.config import IS_RIZMO
from rizmo.image_codec import JpegImageCodec
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import FaceDetection, FaceDetections
from rizmo.nodes.messages_py36 import Box, Detection, Detections
from rizmo.nodes.topics import Topic
from rizmo.py36.client import Py36Client
from rizmo.signal import graceful_shutdown_on_sigterm

Image = np.ndarray


class ObjectDetector:
    @abstractmethod
    def get_objects(self, image: Image) -> list[Detection]:
        ...


class HuggingFaceDetector(ObjectDetector):
    def __init__(self, model, image_processor, threshold: float, allow_labels: Optional[set[str]]):
        self.model = model
        self.image_processor = image_processor
        self.threshold = threshold
        self.allow_labels = allow_labels

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            model_cls,
            image_processor_cls,
            threshold: float = 0.8,
            allow_labels: set[str] = None,
            device: Union[str, torch.device] = None,
    ) -> 'HuggingFaceDetector':
        model = model_cls.from_pretrained(model_name)
        image_processor = image_processor_cls.from_pretrained(model_name)

        if device is None:
            device = 'cuda'  # if torch.cuda.is_available() else 'cpu'

        model = model.to(device)

        return cls(model, image_processor, threshold, allow_labels)

    def get_objects(self, image: Image) -> list[Detection]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(image)

        inputs = self.image_processor(image, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.threshold,
        )[0]

        boxes = results['boxes'].detach()
        boxes = boxes.round_().to(torch.uint16)
        boxes = boxes.cpu().numpy()

        objects = []
        for confidence, label, box in zip(
                results['scores'].cpu().detach().numpy(),
                results['labels'].cpu().detach().numpy(),
                boxes,
        ):
            label = self.model.config.id2label[label]
            if self.allow_labels and label not in self.allow_labels:
                continue

            confidence = confidence.item()
            x_min, y_min, x_max, y_max = [int(it) for it in box]
            box = Box(x_min, y_min, width=x_max - x_min, height=y_max - y_min)
            objects.append(Detection(label, confidence, box))

        return objects


class UltralyticsDetector(ObjectDetector):
    def __init__(
            self,
            model: ultralytics.YOLO,
            conf: float,
    ):
        self.model = model
        self.conf = conf

    @classmethod
    def from_pretrained(
            cls,
            model_name: str,
            conf: float,
    ) -> 'UltralyticsDetector':
        model = ultralytics.YOLO(model_name)
        return cls(model, conf)

    def get_objects(self, image: Image) -> list[Detection]:
        result = self.model(image, conf=self.conf)[0]
        return [self._to_detection(box) for box in result.boxes]

    def _to_detection(self, ul_box) -> Detection:
        return Detection(
            self._get_label(ul_box),
            self._get_confidence(ul_box),
            self._get_box(ul_box),
        )

    def _get_label(self, ul_box) -> str:
        label_idx = ul_box.cls.item()
        return self.model.names[label_idx]

    def _get_confidence(self, ul_box) -> float:
        return ul_box.conf.item()

    def _get_box(self, ul_box) -> Box:
        xyxy = ul_box.xyxy.cpu().numpy()[0]
        xyxy = [round(it) for it in xyxy]
        return Box(
            x=xyxy[0],
            y=xyxy[1],
            width=xyxy[2] - xyxy[0],
            height=xyxy[3] - xyxy[1],
        )


class JetsonDetectNetDetector(ObjectDetector):
    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            py36_client: Py36Client,
            downsample: int = 2,
    ):
        self.loop = loop
        self.py36_client = py36_client
        self.downsample = downsample

    def get_objects(self, image: Image) -> list[Detection]:
        image = image[::self.downsample, ::self.downsample]

        objects = asyncio.run_coroutine_threadsafe(
            self.py36_client.detect(image),
            self.loop,
        ).result()

        for obj in objects:
            obj.box.x *= self.downsample
            obj.box.y *= self.downsample
            obj.box.width *= self.downsample
            obj.box.height *= self.downsample

        return objects


async def main(args: Namespace):
    logging.basicConfig(level=args.log)

    async with await build_node_from_args(args=args) as node:
        await _main(node)


async def _main(node) -> None:
    obj_det_topic = node.get_topic(Topic.OBJECTS_DETECTED)
    faces_detected_topic = node.get_topic(Topic.FACES_DETECTED)

    if IS_RIZMO:
        obj_detector = JetsonDetectNetDetector(
            loop=asyncio.get_event_loop(),
            py36_client=Py36Client.build(),
        )
    else:
        # obj_detector = HuggingFaceDetector.from_pretrained(
        #     # 'facebook/detr-resnet-50', DetrForObjectDetection, DetrImageProcessor,
        #     # 'facebook/detr-resnet-101', DetrForObjectDetection, DetrImageProcessor,
        #     'hustvl/yolos-tiny', YolosForObjectDetection, YolosImageProcessor,
        #     # 'hustvl/yolos-small', YolosForObjectDetection, YolosImageProcessor,
        #     # allow_labels={'person', 'cat'},
        # )

        obj_detector = UltralyticsDetector.from_pretrained(
            'yolo11n.pt',
            # 'yolo11x.pt',
            conf=.5,
        )

    codec = JpegImageCodec()

    @obj_det_topic.depends_on_listener()
    async def handle_image_raw(topic, data):
        timestamp, camera_index, image = data
        image_size = image.shape[1], image.shape[0]
        objects = await asyncio.to_thread(obj_detector.get_objects, image)
        detections = Detections(timestamp, image_size, objects)
        await obj_det_topic.send(detections)
        await send_faces(timestamp, image, image_size, detections)

    @obj_det_topic.depends_on_listener()
    async def handle_image_compressed(topic, data):
        timestamp, camera_index, image_bytes = data
        image, image_size, objects = await asyncio.to_thread(get_objects_from_compressed, image_bytes)
        detections = Detections(timestamp, image_size, objects)
        await obj_det_topic.send(detections)
        await send_faces(timestamp, image, image_size, detections)

    def get_objects_from_compressed(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int], list[Detection]]:
        image = codec.decode(image_bytes)
        image_size = image.shape[1], image.shape[0]
        return image, image_size, obj_detector.get_objects(image)

    async def send_faces(
            timestamp: float,
            image: np.ndarray,
            image_size: tuple[int, int],
            detections: Detections,
    ) -> None:
        faces = [obj for obj in detections.objects if obj.label == 'face']

        if not faces or not await faces_detected_topic.has_listeners():
            return

        face_imgs = (extract_image_fragment(image, face.box) for face in faces)

        faces = [
            FaceDetection(
                face_img,
                confidence=face.confidence,
                box=face.box,
            ) for face, face_img in zip(faces, face_imgs)
        ]

        faces = FaceDetections(timestamp, image_size, faces)
        await faces_detected_topic.send(faces)

    def extract_image_fragment(
            image: np.ndarray,
            box: Box,
            # The face recognition model will resize images to 112x112
            max_size: int = 112,
    ) -> np.ndarray:
        image = image[box.y:box.y + box.height, box.x:box.x + box.width, :]

        h, w, _ = image.shape
        if h <= max_size and w <= max_size:
            return image

        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if IS_RIZMO:
        await node.listen(Topic.NEW_IMAGE_RAW, handle_image_raw)
    else:
        await node.listen(Topic.NEW_IMAGE_COMPRESSED, handle_image_compressed)

    await node.forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
