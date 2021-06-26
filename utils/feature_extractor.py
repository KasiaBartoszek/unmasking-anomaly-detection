from typing import Tuple
import keras.backend as K
import numpy as np
import cv2
from classifiers.VGG16 import VGG16Convolutions
from modules.image_processing import slice, prepare_frames_for_vgg, normalized_gradient, combine
from math import floor
from entities.constants import INDICE_SIZE, MOTION_FEATURE_FRAME_SIZE, SPATIAL_BLOCK_BIN_FRAME_SIZE


class FeatureExtractor:

    def extract_motion_features(self, normal_frames: list, abnormal_frames: list) -> (list, list):
        print(f'Extracting motion features from {len(normal_frames) + len(abnormal_frames)} frames')
        resized_normal_frames = [cv2.resize(frame, MOTION_FEATURE_FRAME_SIZE) for frame in normal_frames]
        resized_abnormal_frames = [cv2.resize(frame, MOTION_FEATURE_FRAME_SIZE) for frame in abnormal_frames]
        normal_frames_indices = [slice(frame, INDICE_SIZE) for frame in resized_normal_frames]
        abnormal_frames_indices = [slice(frame, INDICE_SIZE) for frame in resized_abnormal_frames]

        spatial_cubes_normal = []
        for package in self.__prepare_indices_packages(normal_frames_indices):
            spatial_cubes_normal.append([normalized_gradient(spatial_cube) for spatial_cube in package])

        spatial_cubes_abnormal = []
        for package in self.__prepare_indices_packages(abnormal_frames_indices):
            spatial_cubes_abnormal.append([normalized_gradient(spatial_cube) for spatial_cube in package])

        normal_bins = self.__divide_spatial_cubes_into_bins(spatial_cubes_normal, SPATIAL_BLOCK_BIN_FRAME_SIZE,
                                                            MOTION_FEATURE_FRAME_SIZE, INDICE_SIZE)
        abnormal_bins = self.__divide_spatial_cubes_into_bins(spatial_cubes_abnormal, SPATIAL_BLOCK_BIN_FRAME_SIZE,
                                                            MOTION_FEATURE_FRAME_SIZE, INDICE_SIZE)

        return normal_bins, abnormal_bins

    def __prepare_indices_packages(self, indices: list, size=5):
        size_package = []
        number_of_packages = len(indices[0])
        for index in range(floor(len(indices) / size)):
            packages = []
            for indice_number in range(number_of_packages):
                    packages.append([indices[index + i][indice_number] for i in range(size)])
            size_package.append(packages)

        return size_package

    def __divide_spatial_cubes_into_bins(self, spatial_cubes: list, size: Tuple, resized_frame_size: Tuple, cube_size: Tuple, cube_depth=5) -> list:
        cube_height, cube_width = cube_size
        block_height, block_width = size
        resized_frame_height, resized_frame_width = resized_frame_size
        number_of_sections_y = int(resized_frame_height / block_height)
        number_of_sections_x = int(resized_frame_width / block_width)
        amount_of_cubes_y = int(block_height / cube_height)
        amount_of_cubes_x = int(block_width / cube_width)

        bins = []
        for spatial_cube_index in range(len(spatial_cubes)):
            for depth_level in range(cube_depth):
                frame_blocks = []
                for section_y in range(number_of_sections_y):
                    for section_x in range(number_of_sections_x):
                        blocks = []
                        for y in range(amount_of_cubes_y):
                            for x in range(amount_of_cubes_x):
                                blocks.append(spatial_cubes[spatial_cube_index][y + amount_of_cubes_y * section_y + x + section_x * amount_of_cubes_x][depth_level])
                        frame_blocks.append(blocks)
                bins.append(frame_blocks)

        return bins

    def extract_appearance_features(self, normal_frames: list, abnormal_frames: list) -> (np.ndarray, np.ndarray):
        vgg16_handler = VGG16Convolutions()
        model = vgg16_handler.get_model()

        final_conv_layer = vgg16_handler.get_output_layer(model, "block5_conv3")

        prepared_normal_frames = prepare_frames_for_vgg(normal_frames)
        prepared_abnormal_frames = prepare_frames_for_vgg(abnormal_frames)

        get_output = K.function([model.layers[0].input],
                                [final_conv_layer.output,
                                 model.layers[-1].output])

        outputs_normal, _ = get_output([prepared_normal_frames])
        outputs_abnormal, _ = get_output([prepared_abnormal_frames])

        return outputs_normal, outputs_abnormal




