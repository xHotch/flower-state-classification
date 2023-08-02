import os
from typing import List, Tuple
import numpy as np
import ultralytics
from ultralytics import YOLO, RTDETR
from flower_state_classification.cv.models.modeltypes import Detector
from flower_state_classification.data.boundingbox import BoundingBox


class UltralyticsDetector(Detector):
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

    def predict(self, frames: List[np.array]) -> List[Tuple[BoundingBox, str]]:
        for frame in frames:
            persist = not self.reset_tracker
            result = self.model.track(source=frames, conf=self.threshold, persist=persist, verbose=False)
            self.reset_tracker = False
            names = result[0].names
            boxes = result[0].boxes.cpu().numpy()
            # return [(BoundingBox.from_relative(boxes.xyxy[i], boxes.conf[i]),names[int(boxes.cls[i])]) for i in range(len(boxes))]
            for i, box in enumerate(boxes):
                track_id = boxes.id[i] if box.is_track else -1
                yield (BoundingBox.from_relative(boxes.xyxy[i], boxes.conf[i]), track_id)
