from typing import Tuple

import numpy as np
from flower_state_classification.input.source import Source
import cv2


class WebcamSource(Source):

    # create a opencv video capture object
    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        return self.video_capture.read()
