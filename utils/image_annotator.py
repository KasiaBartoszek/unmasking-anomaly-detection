from typing import Tuple
from math import floor
import numpy as np
import cv2
from entities.constants import ANNOTATIONS_IMAGE_PATH

class ImageAnnotator:

    def __init__(self, images_size: Tuple):
        self.__images_size = images_size
        self.__annotation = cv2.imread(ANNOTATIONS_IMAGE_PATH)

    def annotate_frames(self, frames: list, bins_results: list, bin_size: Tuple, threshold: float) -> list:
        frames = [cv2.resize(frame, self.__images_size) for frame in frames]
        results_per_frame = self.__get_frames_corresponding_results(frames, bins_results)
        annotated_frames = []
        for frame_index in range(len(frames)):
            annotation_indexes = self.__get_bins_indexes_to_annotate(results_per_frame[frame_index], threshold)
            annotated_frames.append(self.__annotate_single_frame(frames[frame_index], annotation_indexes, bin_size))

        return annotated_frames

    def __get_frames_corresponding_results(self, frames: list, bins_results: list) -> list:
        results_per_frame = int(len(bins_results) / len(frames))
        results = []
        for frame_index in range(len(frames)):
            frame_results = []
            for bin_index in range(results_per_frame):
                frame_results.append(bins_results[frame_index * results_per_frame + bin_index])
            results.append(frame_results)

        return results

    def __get_bins_indexes_to_annotate(self, results: list, threshold: float) -> list:
        indexes = []
        for index in range(len(results)):
            if results[index] >= threshold:
                indexes.append(index)

        return indexes

    def __annotate_single_frame(self, frame: np.ndarray, indexes: list, bin_size: Tuple) -> np.ndarray:
        indices_max_y = floor(self.__images_size[0] / bin_size[0])
        indices_max_x = floor(self.__images_size[1] / bin_size[1])

        for annotation_index in indexes:
            y = floor(annotation_index / indices_max_y) * bin_size[0]
            x = floor(annotation_index % indices_max_x) * bin_size[1]
            frame = self.__draw_annotation(frame, (x, y), bin_size)

        return frame

    def __draw_annotation(self, frame: np.ndarray, coords: Tuple, square_size: Tuple) -> np.ndarray:
        x, y = coords
        w, h = square_size
        sub_img = frame[y:y + h, x:x + w]

        if self.__annotation is None:
            self.__annotation = 255 * np.ones(sub_img.shape, np.uint8)
        else:
            self.__annotation = cv2.resize(self.__annotation, square_size)

        res = cv2.addWeighted(sub_img, 0.5, self.__annotation, 0.5, 1.0)

        frame[y:y + h, x:x + w] = res

        return frame

