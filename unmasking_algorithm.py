from classifiers.logistic_regression import LogisticRegressionClassifier
import numpy as np

class UnmaskingAlgorithm:

    __logistic_regression = LogisticRegressionClassifier()

    def __init__(self, normal_motion_features: list, abnormal_motion_features: list):
        self.__normal_motion_features = normal_motion_features
        self.__abnormal_motion_fetaures = abnormal_motion_features

    def run(self, k: int, m: int):
        normal_motion_features = self.__diminish_spatial_bins(self.__normal_motion_features)
        abnormal_motion_features = self.__diminish_spatial_bins(self.__abnormal_motion_fetaures)
        all_results = []
        #running unmasking k times
        for iteration in range(k):
            print(f'Loop {iteration + 1}/{k} started')
            self.__logistic_regression = LogisticRegressionClassifier()
            results = self.__logistic_regression.train_and_classify(normal_motion_features, abnormal_motion_features)
            all_results.append(results)

            #removing m / 2 top 'normal' elements
            for _ in range(int(m / 2)):
                maximal_value = results.max(initial=0)
                index = np.where(results == maximal_value)[0][0]
                if index < len(normal_motion_features):
                    normal_motion_features.pop(index)
                else:
                    abnormal_motion_features.pop(index - len(normal_motion_features))
                results = np.delete(results, index, axis=0)

            # removing m / 2 top 'abnormal' elements
            for _ in range(int(m / 2)):
                maximal_value = results.max(initial=0)
                index = np.where(results == maximal_value)[1][0]
                if index < len(normal_motion_features):
                    normal_motion_features.pop(index)
                else:
                    abnormal_motion_features.pop(index - len(normal_motion_features))
                results = np.delete(results, index, axis=0)

        return all_results

    # getting the abnormality scores
    def get_final_scores(self, bins) -> list:
        diminished_bins = self.__diminish_spatial_bins(bins)
        return [score[1] for score in self.__logistic_regression.classify(diminished_bins)]

    def __diminish_spatial_bins(self, spatial_bins: list):
        diminished_bins = []
        for spatial_bin_index in range(len(spatial_bins)):
            for section_index in range(len(spatial_bins[spatial_bin_index])):
                for index in range(len(spatial_bins[spatial_bin_index][section_index])):
                    diminished_bins.append(spatial_bins[spatial_bin_index][section_index][index])

        return diminished_bins



