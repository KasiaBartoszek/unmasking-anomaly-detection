import cv2
import glob


class FrameLoader:

    def load_frames_from_dir(self, path: str, extension: str) -> list:
        cv_img = []
        for img in glob.glob(f"{path}/**/*.{extension}", recursive=True):
            n = cv2.imread(img)
            cv_img.append(n)

        return cv_img

    #batch_size is the w parameter from paper
    def divide_frames_into_batches(self, frames: list, batch_size = 10) -> (list, list):
        normal = []
        abnormal = []
        for index in range((int)(len(frames)/(batch_size*2))):
            for i in range(batch_size):
                normal.append(frames[index + i])
                abnormal.append(frames[index + batch_size + i])

        return normal, abnormal
