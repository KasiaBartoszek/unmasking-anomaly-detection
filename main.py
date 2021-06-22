from frame_loader import FrameLoader
from feature_extractor import FeatureExtractor
import argparse
import cv2


def run(args):
    frame_loader = FrameLoader()
    feature_extractor = FeatureExtractor()
    frames = frame_loader.load_frames_from_dir(args.path, 'tif')
    normal_frames, abnormal_frames = frame_loader.divide_frames_into_batches(frames)
    appearance_normal, appearance_abnormal = feature_extractor.extract_appearance_features(normal_frames, abnormal_frames)
    cv2.imshow('Normal', appearance_normal[0])
    cv2.waitKey()
    cv2.imshow('Abnormal', appearance_abnormal[0])
    cv2.waitKey()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to images dir')
    args = parser.parse_args()

    run(args)

