import glob
from typing import Tuple

import numpy as np
import cv2

from flower_state_classification.input.source import Source


class ImageFolderSource(Source):
    """
    This class is used to read images from a folder.
    """

    def __init__(self, path: str):
        self.path = path
        self.file_iterator = glob.iglob(path + "/**/*.jpg", recursive=True) # Filter for jpg files
        self.frame_count = len(list(glob.iglob(path + "/**/*.jpg", recursive=True))) # Count number of frames
        self._width = 640 # default values
        self._height = 480 # default values

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Returns the next frame from the source
        """
        filename = next(self.file_iterator, None)
        if filename is None:
            return False, None
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        self._height = image.shape[0]
        self._width = image.shape[1]
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return True, image

    def get_framecount(self) -> int:
        return self.frame_count
    
    def width(self) -> int:
        return self._width
    
    def height(self) -> int:
        return self._height

    def __str__(self) -> str:
        return f"ImageFolderSource({self.path})"
