from typing import Tuple

import numpy as np
from flower_state_classification.input.source import Source
import cv2


class WebcamSource(Source):

    # create a opencv video capture object
    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)
        self.framecount = 0

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        self.framecount += 1
        ret, image = self.video_capture.read()
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ret, image
    
    def get_framecount(self):
        return self.framecount
