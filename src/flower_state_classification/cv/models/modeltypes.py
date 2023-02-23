from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from flower_state_classification.data.boundingbox import BoundingBox

class Classifier(ABC):
    """
    Abstract class for all classifiers, i.e. models that classify images or part of images.
    """
    @abstractmethod
    def predict(frame: np.array) -> str: #Label of the image
        pass

    
class Detector(ABC):
    """
    Abstract class for all detectors, i.e. models that calculate Boundingboxes for images.
    """
    @abstractmethod
    def predict(frame: np.array) -> List[Tuple[BoundingBox, str]]: #List of Boundingboxes and their corresponding labels
        pass
