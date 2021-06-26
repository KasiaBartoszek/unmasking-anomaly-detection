import cv2
from math import floor

class VideoSplitter:

    def split_video_into_frames(self, video_path: str, number_of_frames=200) -> list:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video loaded with {total_frames} frames')


        if total_frames < number_of_frames:
            number_of_frames = total_frames

        step = floor(total_frames / number_of_frames)
        return [self.__extract_frame(video, index * step) for index in range(number_of_frames)]

    def __extract_frame(self, video: cv2.VideoCapture, index: int):
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        result, frame = video.read()

        return frame if result else None
