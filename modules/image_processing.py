from typing import Tuple
import numpy as np
from math import floor
import cv2


def slice(frame: np.ndarray, size: Tuple) -> list:
    indice_height, indice_width = size
    frame_height, frame_width = frame.shape
    rows, columns = floor(frame_height / indice_height), floor(frame_width / indice_width)

    indices = []
    for i in range(0, rows):
        for j in range(0, columns):
            indices.append(frame[i * indice_height:(i + 1) * indice_height, j * indice_width:(j + 1) * indice_width])

    return indices


def combine(indices: list, image_size: Tuple) -> list:
    pass


def normalized_gradient(spatial_cube: list):
    #generating spatial cube's 3D gradient
    cube_gradient = np.gradient(spatial_cube)
    #normalizing spatial cube gradinent using the L^2 norm
    normalized_cube_gradient = np.linalg.norm(cube_gradient, axis=0)
    return normalized_cube_gradient


def convert_frames_to_grayscale(frames: list) -> list:
    print('Converting frames to grayscale')
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]


def prepare_frames_for_vgg(frames: list) -> list:
    mean_index = int(len(frames) / 2)
    rescaled_frames = [__scale_for_vgg(frame) for frame in frames]
    for index in range(len(rescaled_frames)):
        rescaled_frames[index] -= rescaled_frames[mean_index]

    return rescaled_frames


def __scale_for_vgg(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (224, 224))
