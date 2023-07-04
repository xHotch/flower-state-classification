import gc
import os
from typing import Dict, List
import numpy as np
import cv2
import torch
from flower_state_classification.cv.models.ultralytics_models.ultralytics_models import UltralyticsDetector
from flower_state_classification.cv.optical_flow import DenseOpticalFlowCalculator
from flower_state_classification.data.plant import Plant
from flower_state_classification.cv.models.modeltypes import Classifier, Detector
from flower_state_classification.cv.models.huggingface_models.huggingface import HuggingFaceDetector
from flower_state_classification.cv.models.plantnet_models.plantnet import PlantNet
from flower_state_classification.cv.models.paddlepaddle_models.paddlepaddle import PaddleDetectionDetector
from flower_state_classification.debug.debugoutput import box_label
from flower_state_classification.debug.debugsettings import DebugSettings
from flower_state_classification.input.source import Source

import logging

logger=logging.getLogger(__name__)

class FrameProcessor:
    classifier: Classifier = None
    detector: Detector = None
    classified_plants: Dict[int, List[Plant]] = None
    classified_plants_new: Dict[int, List[Plant]] = None

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
            device = "GPU"
            logger.info("Using GPU")
        else:
            device = "CPU"
            self.use_gpu = False
            logger.info("Using CPU")
        #self.classifier = PlantNet(model_name = "vit_base_patch16_224", use_gpu=self.use_gpu, return_genus=False)
        #self.detector = HuggingFaceDetector("facebook/detr-resnet-50", debug_settings)
        dir = os.path.dirname(os.path.abspath(__file__))
        #self.detector = PaddleDetectionDetector(os.path.join(dir,r"models\paddledetection\rtdetr_hgnetv2_l_6x_plants"),device)
        self.detector = UltralyticsDetector()

        self.classified_plants = {}
        self.classified_plants_new = {}
        self.frames_without_plants = []

    def process_frame(self, frame: np.array, frame_nr: int):
        detected_plants = None
        if self.detector:
            detected_plants = self.detector.predict([frame])
        # Classify the image using the classifier
        # TODO currently classifies all detected objects, not only plants
        if detected_plants:
            for bbox, label in detected_plants:
                plant_frame = frame.copy()
                plant_frame = bbox.cut_frame(plant_frame)
                logger.debug(f"Object detector detected label: {label} at location: {bbox} with confidence: {bbox.score}")

                # Get Optical Flow tracker
                is_new_plant = True
                current_plant = None

                if self.classified_plants_new and frame_nr-1 in self.classified_plants_new:
                    for plant in self.classified_plants_new[frame_nr-1]:
                        if label == plant.id or plant.frame_to_bounding_box[frame_nr-1].overlaps(bbox):
                            plant.frame_to_bounding_box[frame_nr] = bbox
                            current_plant = plant
                            is_new_plant = False
                            break

                if is_new_plant:
                    current_plant = Plant(label, {}, True, None, None, DenseOpticalFlowCalculator())
                    current_plant.frame_to_bounding_box[frame_nr] = bbox
                    if frame_nr not in self.classified_plants_new:
                        self.classified_plants_new[frame_nr] = []
                    self.classified_plants_new[frame_nr].append(current_plant)

                # Calculate Optical Flow
                optical_flow = current_plant.optical_flow_calculator.calculate(plant_frame)
                
                classifier_label = self.classifier.predict(plant_frame) if self.classifier else "No classifier"
                gc.collect()
                logger.debug(f"Classifier detected label: {classifier_label}")

                if self.debug_settings.show_plant_frames:
                    cv2.putText(plant_frame, f"{label}, {classifier_label}, {bbox.score}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("plant_frame", cv2.cvtColor(plant_frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                if self.debug_settings.show_optical_flow and optical_flow is not None:
                    cv2.imshow(f"optical_flow {plant.id}", cv2.cvtColor(optical_flow, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

        else:
            logger.debug(f"Object detector did not detect any plants in frame {frame_nr}")
            self.frames_without_plants.append(frame_nr)

        
        if self.debug_settings.show_bboxes and frame_nr in self.classified_plants_new:
            for plant in self.classified_plants_new[frame_nr]:
                box_label(frame, plant.frame_to_bounding_box[frame_nr], f"{plant.id}, {plant.classifier_label}")

        # Displaying the image if the debug setting is set
        if self.debug_settings.show_frame:
            cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.debug_settings.write_video:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        if self.debug_settings.write_images:
            output_filename = os.path.join(self.image_output_folder, f"frame_{frame_nr}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

