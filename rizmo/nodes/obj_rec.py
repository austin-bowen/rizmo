import asyncio
from abc import abstractmethod
from argparse import Namespace
from typing import Optional, Union

import cv2
import numpy as np
import torch
import ultralytics
from PIL import Image as PILImage
from easymesh import build_mesh_node_from_args
from easymesh.asyncio import forever

from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.signal import graceful_shutdown_on_sigterm
from .image_codec import JpegImageCodec
from .messages import Box, Detection, Detections

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
        label_idx = ul_box.cls.item()
        label = self.model.names[label_idx]

        confidence = ul_box.conf.item()

        xyxy = ul_box.xyxy.cpu().numpy()[0]
        xyxy = [round(it) for it in xyxy]
        box = Box(
            x=xyxy[0],
            y=xyxy[1],
            width=xyxy[2] - xyxy[0],
            height=xyxy[3] - xyxy[1],
        )

        return Detection(
            label,
            confidence,
            box,
        )


class Scaler:
    def __init__(self, factor: int):
        self.factor = factor

    def scale_down_image(self, image: np.ndarray) -> np.ndarray:
        return image if self.factor == 1 else image[::self.factor, ::self.factor, :]

    def scale_up_detections(self, detections: list[Detection]) -> list[Detection]:
        if self.factor == 1:
            return detections

        for det in detections:
            det.box.x *= self.factor
            det.box.y *= self.factor
            det.box.width *= self.factor
            det.box.height *= self.factor

        return detections


async def main(args: Namespace):
    node = await build_mesh_node_from_args(args=args)

    obj_det_topic = node.get_topic_sender('objects_detected')

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

    scaler = Scaler(1)

    codec = JpegImageCodec()

    def get_objects(image_bytes: bytes) -> list[Detection]:
        image = codec.decode(image_bytes)
        scaled_image = scaler.scale_down_image(image)
        objects = obj_detector.get_objects(scaled_image)
        return scaler.scale_up_detections(objects)

    @obj_det_topic.depends_on_listener()
    async def handle_image(topic, data):
        timestamp, camera_index, image_bytes = data

        objects = await asyncio.to_thread(get_objects, image_bytes)

        await obj_det_topic.send(Detections(timestamp, objects))

    await node.listen('new_image', handle_image)

    await forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser('obj-rec')
    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
