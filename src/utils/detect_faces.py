import cv2
from constants import *


def detect_faces(image):
    """
    Detecs faces and return rects
    :param image: input image
    :return: faces rects
    """

    cascade_classifier = cv2.CascadeClassifier(CASC_CLASS_MODEL)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    return faces
