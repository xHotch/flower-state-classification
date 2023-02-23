from typing import List, Tuple
import numpy as np

import torch
from flower_state_classification.cv.models.modeltypes import Detector
from transformers import AutoModelForObjectDetection, AutoImageProcessor

from flower_state_classification.data.boundingbox import BoundingBox

class HuggingFaceDetector(Detector):
    def __init__(self, model_name, debug_settings, use_gpu=False):
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
    
    def predict(self, frame) -> List[Tuple[BoundingBox, str]]:
        input_frame = frame.copy()
        inputs = self.image_processor(images=frame, return_tensors="pt")
        outputs = self.model(**inputs)
        if self.use_gpu:
            final_output = final_output.to("cpu")
        input_size = np.shape(input_frame)[:2]
        target_sizes = torch.tensor([input_size])
        #Target size should be tensor([[height, width]])
        

        results = self.image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes = target_sizes)[
            0
        ]

        return [(BoundingBox.from_coco(box, target_sizes,score), self.model.config.id2label[label.item()]) for score, label, box in zip(results["scores"], results["labels"], results["boxes"])]

    
    def __str__(self):
        return f"HuggingFaceDetector({self.model_name})" 