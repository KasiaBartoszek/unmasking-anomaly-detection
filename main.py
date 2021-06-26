from utils.frame_loader import FrameLoader
from utils.feature_extractor import FeatureExtractor
import argparse
from unmasking_algorithm import UnmaskingAlgorithm
import cv2
from utils.image_annotator import ImageAnnotator
from utils.video_splitter import VideoSplitter
from modules.image_processing import convert_frames_to_grayscale
from entities.constants import INDICE_SIZE, MOTION_FEATURE_FRAME_SIZE


def run(args):
    frame_loader = FrameLoader()
    video_splitter = VideoSplitter()

    if args.path:
        frames_original = frame_loader.load_frames_from_dir(args.path, 'tif')
        frames = convert_frames_to_grayscale(frames_original)
        #frames = frames_original
    elif args.video_path:
        frames_original = video_splitter.split_video_into_frames(args.video_path)
        frames = convert_frames_to_grayscale(frames_original)
    else:
        return

    feature_extractor = FeatureExtractor()
    normal_frames, abnormal_frames = frame_loader.divide_frames_into_batches(frames)
    motion_normal_bins, motion_abnormal_bins = feature_extractor.extract_motion_features(normal_frames, abnormal_frames)
    # appearance_normal, appearance_abnormal = feature_extractor.extract_appearance_features(normal_frames, abnormal_frames)
    unmasking_algorithm = UnmaskingAlgorithm(motion_normal_bins, motion_abnormal_bins)
    results = unmasking_algorithm.run(args.k, args.m)
    final_scores = unmasking_algorithm.get_final_scores(motion_normal_bins + motion_abnormal_bins)
    #
    # normal_results, abnormal_results = [result[:, 0] for result in results], [result[:, 1] for result in results]
    # normal_results = [np.mean(value) for value in normal_results]
    # abnormal_results = [np.mean(value) for value in abnormal_results]
    #
    # plt.plot(normal_results, label='Normal')
    # plt.plot(abnormal_results, label='Abnormal')
    # plt.show()
    image_annotator = ImageAnnotator(MOTION_FEATURE_FRAME_SIZE)
    annotated_frames = image_annotator.annotate_frames(frames_original, final_scores, INDICE_SIZE, 0.97)

    for frame_index in range(len(annotated_frames)):
        cv2.imshow(f'Frame {frame_index}', cv2.resize(annotated_frames[frame_index], (400, 400)))
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to images dir')
    parser.add_argument('--video_path', type=str, help='Path to a single video')
    parser.add_argument('--k', type=int, help='Number of framework iterations')
    parser.add_argument('--m', type=int, help='Number of discarded yop features')
    args = parser.parse_args()

    run(args)

