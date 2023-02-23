import os
import numpy as np
import cv2
import torch

from flower_state_classification.data.plant import Plant
from flower_state_classification.cv.models.huggingface import HuggingFaceDetector
from flower_state_classification.cv.models.plantnet.plantnet import PlantNet
from flower_state_classification.debug.debugoutput import box_label
from flower_state_classification.input.source import Source
from flower_state_classification.debug.debugsettings import DebugSettings

import logging

logger=logging.getLogger(__name__)

class FrameProcessor:
    def __init__(self, debug_settings: DebugSettings, source: Source):
        self.debug_settings = debug_settings
        if self.debug_settings.write_video:
            #TODO use max frame size instead of 640x480
            self.video_writer = cv2.VideoWriter(f"{self.debug_settings.output_folder}/output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
        if self.debug_settings.write_images:
            self.image_output_folder = os.path.join(self.debug_settings.output_folder,"frames")
            os.makedirs(self.image_output_folder, exist_ok=True)
        if torch.cuda.is_available():
            self.use_gpu = True
            logger.info("Using GPU")
        else:
            self.use_gpu = False
            logger.info("Using CPU")
        self.classifier = PlantNet(model_name = "resnet18", use_gpu=self.use_gpu, return_genus=False)
        self.detector = HuggingFaceDetector("facebook/detr-resnet-50", debug_settings)

        self.detected_plants = {}
        self.frames_without_plants = []

    def process_frame(self, frame: np.array, frame_nr: int):
        detected_plants = self.detector.predict(frame)
        # Classify the image using the classifier
        # TODO currently classifies all detected objects, not only plants
        if detected_plants and self.classifier:
            for bbox, label in detected_plants:
                plant_frame = frame.copy()
                plant_frame = bbox.cut_frame(plant_frame)
                logger.debug(f"Object detector detected label: {label} at location: {bbox} with confidence: {bbox.score}")
                classifier_label = self.classifier.predict(plant_frame)
                logger.debug(f"Classifier detected label: {classifier_label}")

                if self.debug_settings.show_plant_frames:
                    cv2.putText(plant_frame, f"{label}, {classifier_label}, {bbox.score}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("plant_frame", cv2.cvtColor(plant_frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                if frame_nr not in self.detected_plants:
                    self.detected_plants[frame_nr] = []
                self.detected_plants[frame_nr].append(Plant(None,frame_nr,bbox,True,classifier_label))
        else:
            logger.debug(f"Object detector did not detect any plants in frame {frame_nr}")
            self.frames_without_plants.append(frame_nr)

        if self.debug_settings.show_bboxes:
            for plant in self.detected_plants[frame_nr]:
                box_label(frame, plant.bounding_box, plant.label)

        # Displaying the image if the debug setting is set
        if self.debug_settings.show_frame:
            cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.debug_settings.write_video:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        if self.debug_settings.write_images:
            output_filename = os.path.join(self.image_output_folder, f"frame_{frame_nr}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

