from typing import Tuple
import glob
import numpy as np
import cv2
from flower_state_classification.input.source import Source


class VideoFileSource(Source):
    def __init__(self, path):
        self.path = path
        self.video_capture = cv2.VideoCapture(path)

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        ret, image = self.video_capture.read()
        if ret and len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ret, image

    def get_framecount(self):
        return self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)