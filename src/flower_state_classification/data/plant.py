from dataclasses import dataclass, field
from typing import Dict, List
from flower_state_classification.cv.optical_flow import SparseOpticalFlowCalculator

from flower_state_classification.data.boundingbox import BoundingBox


@dataclass
class Plant:
    """
    Data class for a plant.
    """
    id: str # Unique identifier for the plant
    frame_to_bounding_box: Dict[int, BoundingBox] # Dictionary that maps frame numbers to bounding boxes
    is_healthy: bool # Boolean to indicate if the plant is healthy (needs_water) at the current frame that is processed
    classifier_label: str # Label given by the classifier
    detector_label: str # Label given by the detector that detected the plant
    optical_flow_calculator: SparseOpticalFlowCalculator
    unhealthy_frames: List[int] = field(default_factory=list) # List of frames where the plant is classified as unhealthy (needs_water)
