from dataclasses import dataclass, field
from typing import Dict, List
from flower_state_classification.cv.optical_flow import DenseOpticalFlowCalculator

from flower_state_classification.data.boundingbox import BoundingBox
from flower_state_classification.data.inputdata import InputData


@dataclass
class Plant:
    id: str
    frame_to_bounding_box: Dict[int, BoundingBox]
    is_healthy: bool
    classifier_label: str
    detector_label: str
    optical_flow_calculator: DenseOpticalFlowCalculator
    unhealthy_frames: List[int] = field(default_factory=list)
