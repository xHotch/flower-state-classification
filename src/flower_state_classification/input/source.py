from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Source(ABC):
    @abstractmethod
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Returns the next frame from the source and a boolean indicating if the frame was read successfully.
        """
        pass

    @abstractmethod
    def get_framecount(self) -> int:
        """
        Returns the total number of frames in the source. 
        When the source is a live stream, returns the current frame number.
        """
        pass

    @abstractmethod
    def width(self) -> int:
        """
        Returns the width of the frames in the source
        """
        pass

    @abstractmethod
    def height(self) -> int:
        """
        Returns the height of the frames in the source
        """
        pass