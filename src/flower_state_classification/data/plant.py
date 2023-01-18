from dataclasses import dataclass

from flower_state_classification.data.boundingbox import BoundingBox
from flower_state_classification.data.inputdata import InputData


@dataclass
class Plant:
    """
    Represents a plant in a frame

    At this point, each plant is represented by a bounding box describing the plant for a single frame.
    In the future, we would like to be able to track plants across frames, so that we can track the health of a plant over time.
    """

    input_data: InputData
    frame: int
    bounding_box: BoundingBox
    is_healthy: bool
