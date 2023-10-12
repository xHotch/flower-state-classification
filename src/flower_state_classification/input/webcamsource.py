from typing import Tuple

import numpy as np
from flower_state_classification.input.source import Source
import cv2

import logging

logger = logging.getLogger(__name__)

class WebcamSource(Source):
    """
    Source that reads frames from a webcam.
    """

    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)
        self.framecount = 0
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        self.framecount += 1
        ret, image = self.video_capture.read()
        if not ret:
            logger.warning("Could not read frame from Webcam")
        if ret and len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ret, image

    def get_framecount(self) -> int:
        return self.framecount

    def width(self) -> int:
        return self._width
    
    def height(self) -> int:
        return self._height