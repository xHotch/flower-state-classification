from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    x2: int
    y2: int
    
    image_width: int
    image_height: int

    score: float = 0.0

    @classmethod
    def from_coco(cls, coco_bbox, size, score):
        image_height, image_width = int(size[0][0]), int(size[0][1])
        box = [int(i) for i in coco_bbox.tolist()]
        if any(box<0 for box in box):
            logger.warning(f"Negative values in bounding box: {box}")
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
        x2 = box[0]+ box[2]
        if x2 > image_width:
            logger.warning(f"Bounding box exceeds image width: {x2} > {image_width}")
            box[2] = image_width - box[0]
        y2 = box[1]+ box[3]    
        if y2 > image_height:
            logger.warning(f"Bounding box exceeds image height: {y2} > {image_height}")
            box[3] = image_height - box[1]
        
        return cls(box[0], box[1], box[2], box[3], x2, y2, image_width, image_height, score)
    
    def cut_frame(self, frame):
        return frame[self.y:self.y+self.height, self.x:self.x+self.width]
    
        
    