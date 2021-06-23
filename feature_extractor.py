import keras.backend as K
import numpy as np
import cv2
from classifiers.VGG16 import VGG16Convolutions


class FeatureExtractor:

    def extract_appearance_features(self, normal_frames: list, abnormal_frames: list) -> (np.ndarray, np.ndarray):
        vgg16_handler = VGG16Convolutions()
        model = vgg16_handler.get_model()

        final_conv_layer = vgg16_handler.get_output_layer(model, "block5_conv3")

        prepared_normal_frames = self.__prepare_frames_for_vgg(normal_frames)
        prepared_abnormal_frames = self.__prepare_frames_for_vgg(abnormal_frames)

        get_output = K.function([model.layers[0].input],
                                [final_conv_layer.output,
                                 model.layers[-1].output])

        outputs_normal, _ = get_output([prepared_normal_frames])
        outputs_abnormal, _ = get_output([prepared_abnormal_frames])

        return outputs_normal, outputs_abnormal

    def __prepare_frames_for_vgg(self, frames: list) -> list:
        mean_index = int(len(frames) / 2)
        rescaled_frames = [self.__scale_for_vgg(frame) for frame in frames]
        for index in range(len(rescaled_frames)):
            rescaled_frames[index] -= rescaled_frames[mean_index]

        return rescaled_frames

    def __scale_for_vgg(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (224, 224))
