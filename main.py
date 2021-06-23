from frame_loader import FrameLoader
from feature_extractor import FeatureExtractor
import argparse
from unmasking_algorithm import UnmaskingAlgorithm
import cv2

def run(args):
    frame_loader = FrameLoader()
    feature_extractor = FeatureExtractor()
    frames = frame_loader.load_frames_from_dir(args.path, 'tif')
    normal_frames, abnormal_frames = frame_loader.divide_frames_into_batches(frames)
    appearance_normal, appearance_abnormal = feature_extractor.extract_appearance_features(normal_frames, abnormal_frames)
    unmasking_algorithm = UnmaskingAlgorithm(normal_frames, abnormal_frames, appearance_normal, appearance_abnormal)
    results = unmasking_algorithm.run(args.k, args.m)
    final_scores = unmasking_algorithm.get_final_scores(frames)

    cv2.imshow(f'{final_scores[20]}', frames[20])
    cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to images dir')
    parser.add_argument('--k', type=int, help='Number of framework iterations')
    parser.add_argument('--m', type=int, help='Number of discarded yop features')
    args = parser.parse_args()

    run(args)

