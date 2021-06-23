from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegressionClassifier:

    def __init__(self):
        self.__classifier = LogisticRegression(random_state=0, max_iter=10000)

    def train_and_classify(self, frames_normal: list, frames_abnormal: list):
        frames_reshaped = self.__reshape(frames_normal + frames_abnormal)

        y = self.__assign_classes(len(frames_normal), 'Normal') + self.__assign_classes(len(frames_abnormal), 'Abnormal')
        self.__classifier = self.__classifier.fit(frames_reshaped, y)
        return self.__classifier.predict_proba(frames_reshaped)

    def __split_frames(self, frames) -> (list, list):
        return frames[:int(len(frames)*0.8)], frames[int(len(frames)*0.8):]

    def __assign_classes(self, amount: int, name: str) -> list:
        return [name for _ in range(amount)]

    def __reshape(self, frames: list) -> list:
        n_samples = len(frames)
        return np.array(frames).reshape((n_samples, -1))

    def classify(self, frames) -> np.ndarray:
        frames_reshaped = self.__reshape(frames)
        return self.__classifier.predict_proba(frames_reshaped)
