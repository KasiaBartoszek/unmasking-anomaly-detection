from frame_loader import FrameLoader
from feature_extractor import FeatureExtractor
import argparse
from unmasking_algorithm import UnmaskingAlgorithm
import cv2
import matplotlib.pyplot as plt

def run(args):
    frame_loader = FrameLoader()
    feature_extractor = FeatureExtractor()
    frames = frame_loader.load_frames_from_dir(args.path, 'tif')
    normal_frames, abnormal_frames = frame_loader.divide_frames_into_batches(frames)
    #appearance_normal, appearance_abnormal = feature_extractor.extract_appearance_features(normal_frames, abnormal_frames)
    unmasking_algorithm = UnmaskingAlgorithm(normal_frames, abnormal_frames, [], [])
    results = unmasking_algorithm.run(args.k, args.m)
    final_scores = unmasking_algorithm.get_final_scores(frames)

    normal_values = []
    abnormal_values = []
    normal_averages = []
    abnormal_averages = []

    for index in range(len(results)):
        for i in range(len(results[index])):
            normal_values.append(results[index][i][0])
            abnormal_values.append(results[index][i][1])
        normal_averages.append(sum(normal_values) / len(normal_values))
        abnormal_averages.append(sum(abnormal_values) / len(abnormal_values))

    plt.plot(normal_averages)
    plt.plot(abnormal_averages)
    plt.show()

    #cv2.imshow(f'{final_scores[20]}', frames[20])
    #cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to images dir')
    parser.add_argument('--k', type=int, help='Number of framework iterations')
    parser.add_argument('--m', type=int, help='Number of discarded yop features')
    args = parser.parse_args()

    run(args)

