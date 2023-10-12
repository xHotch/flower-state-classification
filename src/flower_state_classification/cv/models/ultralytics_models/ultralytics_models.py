import os
from typing import List, Tuple
import numpy as np
import ultralytics
from ultralytics import YOLO, RTDETR
from flower_state_classification.cv.models.modeltypes import Tracker
from flower_state_classification.data.boundingbox import BoundingBox


class UltralyticsDetector(Tracker):
    """
    Use a detector from the Ultralytics library to detect objects in images.

    This is the default detector for the Flower State Classification project.
    Model yolov8_m_openimages_best.pt contains the weights from the best performing model on the combined dataset.
    """
    def __init__(self, model_name: str = "yolov8_m_openimages_best.pt", threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.dir = os.path.dirname(os.path.abspath(__file__))
        if model_name.lower().startswith("yolo"):
            self.model = YOLO(os.path.join(self.dir, model_name))
        elif model_name.lower().startswith("rtdetr"):
            self.model = RTDETR(os.path.join(self.dir, model_name))
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        self.reset_tracker = True

    def predict(self, frame: np.ndarray) -> List[Tuple[BoundingBox, str]]:
        persist = not self.reset_tracker
        result = self.model.track(source=[frame], conf=self.threshold, persist=persist, verbose=False, tracker="botsort.yaml")
        self.reset_tracker = False
        names = result[0].names
        boxes = result[0].boxes.cpu().numpy()

        for i, box in enumerate(boxes):
            track_id = boxes.id[i] if box.is_track else -1
            yield (BoundingBox.from_ultralytics(boxes.xyxy[i], boxes.conf[i], np.shape(frame)), track_id)
