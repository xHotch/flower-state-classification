import glob
from typing import Tuple

import numpy as np
import cv2

from flower_state_classification.input.source import Source


class ImageFolderSource(Source):
    """
    This class is used to read images from a folder.
    """

    def __init__(self, path):
        self.path = path
        self.file_iterator = glob.iglob(path + "/**/*.jpg", recursive=True)
        self.frame_number = len(list(glob.iglob(path + "/**/*.jpg", recursive=True)))
        
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        filename = next(self.file_iterator, None)
        if filename is None:
            return False, None
        return True, cv2.imread(filename)
