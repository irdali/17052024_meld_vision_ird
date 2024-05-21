import cv2
import os
from glob import glob
import numpy as np


def generate_path(filename, mode, base_path, extracted_faces: bool = False):
    if not extracted_faces:
        if 'mp4' not in filename:
            filename += '.mp4'

    if mode not in ['train', 'test', 'dev', 'val']:
        print("Invalide mode specified")
        return None

    if mode == 'val':
        mode = 'dev'

    path = os.path.join(base_path, mode, filename)

    if not os.path.exists(path):
        print("Could not locate the file/folder\
        with specified parameters.")
        return None
    return path


def play_video(filename, mode, base_path, wait_time=40):
    path = generate_path(filename, mode, base_path)
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(filename, frame)

        cv2.imwrite('x.jpg', frame)
        if cv2.waitKey(wait_time) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 1


def get_extracted_faces_video(filename, mode, base_path, ):
    path = generate_path(filename, mode, base_path, extracted_faces=True)

    data = {}
    print(path)

    frames = [os.path.join(path, frame) for frame in os.listdir(path)]

    for frame in frames:
        faces_path = glob('*.jpg')
        frame_no = frame.split('/')[-1]
        faces = []
        for face_path in faces_path:
            faces.append(cv2.imread(face_path))
        face_img = np.vstack(faces)
        cv2.imshow("X", cv2.resize(face_img, (512, 512)))
        data[frame_no] = face_img

        if cv2.waitKey(0) == ord('q'):
            break

    return data
