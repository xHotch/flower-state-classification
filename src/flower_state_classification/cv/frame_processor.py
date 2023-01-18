import numpy as np
import cv2

from flower_state_classification.cv.debug.debugsettings import DebugSettings


class FrameProcessor:
    def __init__(self, debug_settings: DebugSettings):
        self.debug_settings = debug_settings
        ...

    def process_frame(self, frame: np.array):

        if self.debug_settings.show_frame:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
