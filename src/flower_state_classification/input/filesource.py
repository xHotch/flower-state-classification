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
        return self.video_capture.read()
