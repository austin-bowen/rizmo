import asyncio
import logging
from argparse import Namespace
from dataclasses import dataclass

import numpy as np
from rosy import build_node_from_args
from rosy.asyncio import forever

from rizmo.face_rec.face_finder import build_face_finder
from rizmo.face_rec.image_store import MultiImagePerNameFileStore
from rizmo.node_args import get_rizmo_node_arg_parser
from rizmo.nodes.messages import FaceDetections, FaceRecognition, FaceRecognitions
from rizmo.nodes.topics import Topic
from rizmo.signal import graceful_shutdown_on_sigterm

FACE_STORE_ROOT: str = './var/faces'


async def main(args: Namespace) -> None:
    logging.basicConfig(level=args.log)

    @dataclass
    class Cache:
        save_face_name: str | None = None

    cache = Cache()

    face_finder = build_face_finder(
        embedding_model=args.embedding_model,
        face_db_device=args.face_db_device,
        min_similarity=args.min_similarity,
    )

    face_store = MultiImagePerNameFileStore(
        FACE_STORE_ROOT,
        file_type='bmp',
    )

    face_finder.add_faces(face_store.get_all())

    async def handle_faces_detected(topic, face_detections: FaceDetections) -> None:
        face_imgs = [face.image for face in face_detections.faces]

        if cache.save_face_name and len(face_imgs) == 1:
            cache.save_face_name, face_name = None, cache.save_face_name
            await save_face(face_imgs[0], face_name)

        face_recs = await asyncio.to_thread(
            face_finder.find_faces,
            face_imgs,
        )

        face_recs_log = [(str(name), round(sim, 3)) for name, sim in face_recs]
        face_recs_log = sorted(face_recs_log)
        print('Faces recognized:', face_recs_log)

        face_recs = [
            FaceRecognition(name, sim, face.box)
            for (name, sim), face in zip(face_recs, face_detections.faces)
        ]

        face_recs = FaceRecognitions(
            face_detections.timestamp,
            face_detections.image_size,
            face_recs,
        )

        await faces_recognized_topic.send(face_recs)

    async def save_face(face_img: np.ndarray, face_name: str) -> None:
        print(f'Saving face image of size {face_img.shape} with name: {face_name}')
        face_store.add(face_img, face_name)
        face_finder.add_face(face_img, face_name)

    async def handle_face_command(topic, command: dict) -> None:
        action = command['action']
        name = command['name']

        if action == 'add':
            cache.save_face_name = name
        else:
            raise ValueError(f'Invalid action: {action}')

    node = await build_node_from_args(args=args)
    faces_recognized_topic = node.get_topic(Topic.FACES_RECOGNIZED)
    await node.listen(Topic.FACE_COMMAND, handle_face_command)
    await node.listen(Topic.FACES_DETECTED, handle_faces_detected)

    await forever()


def parse_args() -> Namespace:
    parser = get_rizmo_node_arg_parser(__file__)

    parser.add_argument(
        '--embedding-model',
        default='w600k_r50',
        choices=('w600k_r50', 'w600k_mbf'),
        help='Face embedding model to use for face recognition. Default: %(default)s',
    )

    parser.add_argument(
        '--face-db-device',
        default='auto',
        choices=('auto', 'cpu', 'cuda'),
        help='Device to use for face database search. '
             '"auto" will use "cuda" if available, or fall back to "cpu". '
             'Default: %(default)s',
    )

    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.3,
        help='Minimum similarity threshold for face recognition. Default: %(default)s',
    )

    return parser.parse_args()


if __name__ == '__main__':
    graceful_shutdown_on_sigterm()
    asyncio.run(main(parse_args()))
