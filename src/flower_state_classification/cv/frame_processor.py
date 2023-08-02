import gc
import os
from typing import Dict, List
import numpy as np
import cv2
from flower_state_classification.cv.optical_flow import DenseOpticalFlowCalculator, SparseOpticalFlowCalculator
from flower_state_classification.data.plant import Plant
from flower_state_classification.cv.models.modeltypes import Classifier, Detector
from flower_state_classification.debug.debugoutput import box_label
from flower_state_classification.debug.settings import CameraMode, Settings

import logging

from flower_state_classification.util.timer import Timer

logger=logging.getLogger(__name__)

class FrameProcessor:
    classifier: Classifier = None
    detector: Detector = None
    classified_plants: Dict[int, List[Plant]] = None
    classified_plants_new: Dict[int, List[Plant]] = None
    last_frame: np.ndarray = None
    
    def __init__(self, run_settings: Settings):
        self.debug_settings = run_settings
        self.classifier = run_settings.classifier
        self.detector = run_settings.detector

        if self.debug_settings.write_video:
            #TODO use max frame size instead of 640x480
            self.video_writer = cv2.VideoWriter(f"{self.debug_settings.output_folder}/output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
        
        self.image_output_folder = self.debug_settings.image_output_folder
        self.classified_plants = {}
        self.classified_plants_new = {}
        self.frames_without_plants = []
        self.optical_flow_calculator = SparseOpticalFlowCalculator(self.debug_settings)
        self.optical_flow_calculator2 = DenseOpticalFlowCalculator(self.debug_settings)


    def process_frame(self, frame: np.array, frame_nr: int):
        detected_plants = None
        if self.detector:
            with Timer("Detection", logger.debug):
                detected_plants = self.detector.predict([frame])

        # Calculate optical flow on whole frame for static cameras
        optical_flow = None
        #optical_flow_polar = None
        if self.debug_settings.camera_mode is CameraMode.STATIC:
            # optical_flow = self.optical_flow_calculator.calculate(frame)
            optical_flow2 = self.optical_flow_calculator2.calculate(frame)
                    

        # Classify the image using the classifier
        if detected_plants:
            for bbox, label in detected_plants:
                optical_flow = self.optical_flow_calculator.calculate(frame, bbox)
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
                            if frame_nr not in self.classified_plants_new:
                                self.classified_plants_new[frame_nr] = []
                            self.classified_plants_new[frame_nr].append(plant)
                            current_plant = plant
                            is_new_plant = False
                            break

                if is_new_plant:
                    optical_flow_calculator = SparseOpticalFlowCalculator(self.debug_settings) if self.debug_settings.camera_mode is CameraMode.MOVING else self.optical_flow_calculator
                    current_plant = Plant(label, {}, True, None, None, optical_flow_calculator)
                    current_plant.frame_to_bounding_box[frame_nr] = bbox
                    if frame_nr not in self.classified_plants_new:
                        self.classified_plants_new[frame_nr] = []
                    self.classified_plants_new[frame_nr].append(current_plant)

                # Calculate Optical Flow
                green_mask = None
                if self.last_frame is not None:
                    if self.debug_settings.camera_mode is CameraMode.MOVING:
                        optical_flow = current_plant.optical_flow_calculator.calculate(plant_frame)
                    elif self.debug_settings.camera_mode is CameraMode.STATIC:
                        needs_water = current_plant.optical_flow_calculator.classify(optical_flow)
                        self.optical_flow_calculator2.classify(optical_flow2)


                
                classifier_label = self.classifier.predict(plant_frame) if self.classifier else "No classifier"
                gc.collect()
                logger.debug(f"Classifier detected label: {classifier_label}")

                if self.debug_settings.show_plant_frames:
                    cv2.putText(plant_frame, f"{label}, {classifier_label}, {bbox.score}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("plant_frame", cv2.cvtColor(plant_frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                if self.debug_settings.show_optical_flow and optical_flow is not None:
                    output_frame = frame.copy()
                    # optical_arrows = self.optical_flow_calculator.visualize(output_frame, optical_flow)
                    # cv2.imshow("optical_flow_arrows", cv2.cvtColor(optical_arrows, cv2.COLOR_RGB2BGR))
                    #cv2.waitKey(-1)

                if self.debug_settings.plot_optical_flow and optical_flow is not None:
                    if frame_nr % 10 == 0:
                        current_plant.optical_flow_calculator.plot(current_plant.optical_flow_calculator.flow_angles, current_plant.optical_flow_calculator.flow_magnitudes, id = current_plant.id, save = True)
                        current_plant.optical_flow_calculator.plot(current_plant.optical_flow_calculator.total_angles, current_plant.optical_flow_calculator.total_magnitudes, id = str(current_plant.id) + "_total_flow", save = True)

        else:
            logger.debug(f"Object detector did not detect any plants in frame {frame_nr}")
            self.frames_without_plants.append(frame_nr)

        output_frame = frame.copy()
        if self.debug_settings.show_bboxes and frame_nr in self.classified_plants_new:
            for plant in self.classified_plants_new[frame_nr]:
                box_label(output_frame, plant.frame_to_bounding_box[frame_nr], f"{plant.id}, {plant.classifier_label}")
        else:
            logger.debug(f"No bounding boxes to show in frame {frame_nr}")

        # Displaying the image if the debug setting is set
        if self.debug_settings.show_frame:
            cv2.imshow("frame", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        

        if self.debug_settings.write_video:
            self.video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        
        if self.debug_settings.write_images:
            output_filename = os.path.join(self.image_output_folder, f"frame_{frame_nr}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
        
        self.last_frame = frame

