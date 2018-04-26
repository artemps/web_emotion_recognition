import os

import tflearn
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

from constants import *


def define_network():
    """
    Defines CNN architecture
    :return: CNN model
    """

    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.0)
    img_aug.add_random_blur(sigma_max=3.0)

    network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1],
                         data_augmentation=img_aug)

    network = conv_2d(network, 64, 3, activation='relu')
    network = batch_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu')
    network = batch_normalization(network)
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 128, 3, activation='relu')
    network = batch_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu')
    network = batch_normalization(network)
    network = max_pool_2d(network, 2, strides=2)
    network = dropout(network, 0.2)

    network = conv_2d(network, 256, 3, activation='relu')
    network = batch_normalization(network)
    network = conv_2d(network, 256, 3, activation='relu')
    network = batch_normalization(network)
    network = max_pool_2d(network, 2, strides=2)
    network = dropout(network, 0.25)

    network = conv_2d(network, 512, 3, activation='relu')
    network = batch_normalization(network)
    network = conv_2d(network, 512, 3, activation='relu')
    network = batch_normalization(network)
    network = max_pool_2d(network, 2, strides=2)
    network = dropout(network, 0.25)

    network = fully_connected(network, 1024, activation='relu')
    network = batch_normalization(network)
    network = dropout(network, 0.45)

    network = fully_connected(network, 1024, activation='relu')
    network = batch_normalization(network)
    network = dropout(network, 0.45)

    network = fully_connected(network, len(EMOTIONS), activation='softmax')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy')

    model = tflearn.DNN(network, checkpoint_path=os.path.join('/emotion_recognition'),
                        max_checkpoints=1, tensorboard_verbose=0)

    return model


def load_trained_model():
    """
    Loads trained model from save dir
    :return: trained model
    """

    model = define_network()
    model.load(os.path.join(MODEL_DIR, 'emotion_recognizer'))
    return model
