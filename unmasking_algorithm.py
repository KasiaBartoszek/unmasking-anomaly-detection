from classifiers.logistic_regression import LogisticRegressionClassifier
import numpy as np

class UnmaskingAlgorithm:

    __logistic_regression = LogisticRegressionClassifier()

    def __init__(self, normal_frames: list, abnormal_frames: list, motion_features: list, appearence_features: list):
        self.__normal_frames = normal_frames
        self.__abnormal_frames = abnormal_frames
        self.__motion_features = motion_features
        self.__apearence_features = appearence_features

    def run(self, k: int, m: int):
        all_results = []
        #running unmasking k times
        for iteration in range(k):
            self.__logistic_regression = LogisticRegressionClassifier()
            results = self.__logistic_regression.train_and_classify(self.__normal_frames, self.__abnormal_frames)
            all_results.append(results)

            #removing m top 'normal' elements
            for _ in range(m):
                maximal_value = results.max(initial=0)
                index = np.where(results == maximal_value)[0][0]
                if index < len(self.__normal_frames):
                    self.__normal_frames.pop(index)
                else:
                    self.__abnormal_frames.pop(index - len(self.__normal_frames))
                results = np.delete(results, index, axis=0)

        return all_results

    #getting the abnormality scores
    def get_final_scores(self, frames) -> list:
        return [score[1] for score in self.__logistic_regression.classify(frames)]



