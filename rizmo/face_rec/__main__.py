import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from rizmo.face_rec.face_finder import build_face_finder


def main():
    face_detector = FaceAnalysis(
        name='buffalo_sc',
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
        root='/tmp/insightface',
        allowed_modules=[
            'detection',
        ],
        allow_download=True,
    )
    face_detector.prepare(ctx_id=0, det_size=(640, 640))

    similar_face_finder = build_face_finder()

    def get_face_recs(img: np.ndarray) -> list[tuple[str, float]]:
        faces = face_detector.get(img)

        face_imgs = (extract_face(img, face.bbox) for face in faces)
        face_imgs = list(filter(is_valid_face_img, face_imgs))

        return similar_face_finder.find_faces(face_imgs) if face_imgs else []

    video = cv2.VideoCapture(0)
    try:
        while True:
            ret, img = video.read()
            if not ret:
                break

            face_recs = get_face_recs(img)
            face_recs = [(str(name), round(sim, 3)) for name, sim in face_recs]
            face_recs = sorted(face_recs)
            print('Face recognitions:', face_recs)

            time.sleep(1 / 5)
    finally:
        video.release()


def extract_face(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    bbox = bbox.round().astype(int)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]


def is_valid_face_img(face_img: np.ndarray, min_size: int = 10) -> bool:
    h, w = face_img.shape[:2]
    return h >= min_size and w >= min_size


if __name__ == '__main__':
    main()
