from dataclasses import dataclass
import logging

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x: int
    y: int
    x2: int
    y2: int
    
    image_width: int = 1280
    image_height: int = 720

    score: float = 0.0

    @classmethod
    def from_coco(cls, coco_bbox, size, score):
        image_height, image_width = int(size[0][0]), int(size[0][1])
        box = [int(i) for i in coco_bbox.tolist()]
        
        if any(box<0 for box in box):
            logger.warning(f"Negative values in bounding box: {box}")
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])

        if box[2] > image_width:
            logger.warning(f"Bounding box exceeds image width: {box[2]} > {image_width}")
            box[2] = min(image_width, box[2])

        if box[3] > image_height:
            logger.warning(f"Bounding box exceeds image height: {box[3]} > {image_height}")
            box[3] = min(image_height, box[3])
        
        return cls(box[0], box[1], box[2], box[3], image_width, image_height, score)
    
    @classmethod
    def from_relative(cls, box, score, normalized=False):
        x1, y1, x2, y2 = [int(data) for data in box]
        if any(box<0 for box in box):
            logger.warning(f"Negative values in bounding box: {box}")
            x1 = max(0, x1)
            y1 = max(0, y1)
        if normalized:
            return cls(x1*cls.image_width, y1*cls.image_height, x2*cls.image_width, y2*cls.image_height, score=score)
        return cls(x1, y1, x2, y2, score=score)

    def height(self):
        return self.y2-self.y
    
    def width(self):
        return self.x2-self.x

    def area(self):
        return (self.width())*(self.height())
    
    def intersection_area(self, other):
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        return x_overlap * y_overlap

    def union_area(self, other):
        return self.area() + other.area() - self.intersection_area(other)
    
    def overlaps(self, other, threshold = 0.8):
        """
        Returns true if the IOU of this bounding box with the other bounding box is greater than threshold.
        """
        iou = self.intersection_area(other) / self.union_area(other)
        return iou > threshold

    def mask_frame(self, frame, value=0):
        """
        Masks the frame with the bounding box.
        """
        new_frame = np.full(frame.shape, value, dtype=frame.dtype)
        new_frame[self.y:self.y2, self.x:self.x2] = frame[self.y:self.y2, self.x:self.x2]
        return new_frame


    def cut_frame(self, frame):
        return frame[int(self.y):int(self.y2), int(self.x):int(self.x2)]
    
        
    