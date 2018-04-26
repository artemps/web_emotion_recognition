from datetime import datetime
import cv2
import numpy as np

from constants import *
from .detect_faces import detect_faces


def save_img(img, i, emotion, dt):
    """
    Save image to user data dir
    :param img: input image
    :param i: image number
    :param emotion: recognized emotion
    :param dt: datetime
    :return: 
    """

    cv2.imwrite('{}/{}_{}_{}.jpg'.format(USER_DATA_DIR, i, emotion, dt), img)


def crop_faces(image, rects):
    """
    Crops faces from image and resize
    :param image: input image
    :param rects: face rectangles
    :return: array of cropped faces
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces_images = []
    for rect in rects:
        face_image = image[rect[1]:rect[1] + rect[2], rect[0]:rect[0] + rect[3]]
        face_image = cv2.resize(face_image, (IMG_SIZE, IMG_SIZE))
        face_image = face_image.astype('float32') / 255.
        face_image = face_image.reshape([-1, IMG_SIZE, IMG_SIZE, 1])
        faces_images.append(face_image)

    return faces_images


def recognize(model, data):
    """
    Recgontize emotions on model
    :param model: trained model 
    :param data: image data
    :return: face_rects, emotions
    """

    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    face_rects = detect_faces(image)
    images = crop_faces(image, face_rects)

    predictions = []
    for img in images:
        predictions.append(model.predict(img)[0])

    resp = {'Faces': []}
    for i, face in enumerate(face_rects):
        _obj = {'Face Rectangles': {'x': int(face[0]),
                                    'y': int(face[1]),
                                    'width': int(face[2]),
                                    'height': int(face[3])},
                'Emotions': {e: round(float(p), 3)
                             for e, p in zip(EMOTIONS, predictions[i])}}
        resp['Faces'].append(_obj)

        save_img(image, i, np.argmax(predictions[i]), datetime.now())

    return resp
