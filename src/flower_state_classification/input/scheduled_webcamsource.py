import datetime
import time
from typing import Tuple
import cv2
import numpy as np
from flower_state_classification.settings.settings import Settings
from flower_state_classification.input.source import Source


class ScheduledWebcamsource(Source):
    """
    Source that reads from a webcam and uses a simple schedule the wait when reading new frames.
    """
    def __init__(self, start_time: str, end_time: str, cooldown_in_minutes: int) -> None:
        self.video_capture = cv2.VideoCapture(0)
        self.framecount = 0
        self._width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.daily_start_time = datetime.datetime.strptime(start_time, "%H:%M").time()
        self.daily_end_time = datetime.datetime.strptime(end_time, "%H:%M").time()
        self.cooldown_in_minutes = cooldown_in_minutes

        if self.cooldown_in_minutes < 2:
            self._retry_in_seconds = 5
        else:
            self._retry_in_seconds = 60
        self.last_frame_time = None

    def _should_wait(self) -> bool:
        current_time = datetime.datetime.now()
        if current_time.time() < self.daily_start_time or current_time.time() > self.daily_end_time:
            return True
        if not self.last_frame_time:
            return False
        
        next_frame_time = self.last_frame_time + datetime.timedelta(0,self.cooldown_in_minutes * 60)
        if current_time < next_frame_time:
            return True
        return False

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        while self._should_wait():
            time.sleep(self._retry_in_seconds)

        self.last_frame_time = datetime.datetime.now()
        self.framecount += 1
        ret, image = self.video_capture.read()
        if ret and len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ret, image

    def get_framecount(self) -> int:
        return self.framecount

    def width(self) -> int:
        return self._width
    
    def height(self) -> int:
        return self._height