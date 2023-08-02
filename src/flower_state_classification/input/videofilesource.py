from typing import Tuple
import glob
import numpy as np
import cv2
from flower_state_classification.input.source import Source


class VideoFileSource(Source):
    def __init__(self, path: str, max_frames: int = None):
        self.path = path
        self.video_capture = cv2.VideoCapture(path)
        self.max_frames = max_frames
        self.frame_number = 0
        if self.max_frames:
            self.step_size = int(self._get_source_framecount() / self.get_framecount())
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        # Skip frames if max_frames is set
        if self.max_frames:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number * self.step_size)

        ret, image = self.video_capture.read()
        if ret and len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.frame_number += 1
        return ret, image

    def width(self):
        return self._width
    
    def height(self):
        return self._height
    
    def _get_source_framecount(self) -> int:
        return self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def get_framecount(self) -> int:
        video_frames = self._get_source_framecount()
        if self.max_frames:
            return self.max_frames if self.max_frames < video_frames else video_frames
        return video_frames

    def __str__(self) -> str:
        return super().__str__() + f" {self.path}"
