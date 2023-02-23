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
        self.frame_count = len(list(glob.iglob(path + "/**/*.jpg", recursive=True)))
        
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        filename = next(self.file_iterator, None)
        if filename is None:
            return False, None
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return True, image
    
    def get_annotation(self, frame_nr: int):
        return None

    def get_framecount(self):
        return self.frame_count

    def __str__(self):
        return f"ImageFolderSource({self.path})"