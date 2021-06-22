from keras.applications.vgg16 import VGG16
import keras.backend as K
import numpy as np
import cv2


class FeatureExtractor:

    def extract_appearance_features(self, normal_frames: list, abnormal_frames: list) -> (np.ndarray, np.ndarray):
        model = VGG16()
        class_weights = model.layers[-1].get_weights()[0]
        prepared_normal_frames = self.__prepare_frames_for_vgg(normal_frames)
        prepared_abnormal_frames = self.__prepare_frames_for_vgg(abnormal_frames)
        get_output = K.function([model.layers[0].input],
                                [model.layers[-1].output,
                                 model.layers[-1].output])

        outputs_normal, _ = get_output([prepared_normal_frames])
        appearance_features_normal = np.zeros(dtype=np.float32, shape=outputs_normal.shape[1:3])
        target_class = 1
        for i, w in enumerate(class_weights[:, target_class]):
            appearance_features_normal += w * outputs_normal[i, :]

        outputs_abnormal, _ = get_output([prepared_abnormal_frames])
        appearance_features_abnormal = np.zeros(dtype=np.float32, shape=outputs_abnormal.shape[1:3])
        target_class = 1
        for i, w in enumerate(class_weights[:, target_class]):
            appearance_features_abnormal += w * outputs_abnormal[i, :]

        return appearance_features_normal, appearance_features_abnormal


    def __prepare_frames_for_vgg(self, frames: list) -> list:
        mean_index = int(len(frames) / 2)
        rescaled_frames = [self.__scale_for_vgg(frame) for frame in frames]
        for index in range(len(rescaled_frames)):
            rescaled_frames[index] -= rescaled_frames[mean_index]

        return rescaled_frames

    def __scale_for_vgg(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (224, 224))
