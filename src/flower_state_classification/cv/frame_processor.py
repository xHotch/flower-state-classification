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
from flower_state_classification.input.source import Source

from flower_state_classification.util.timer import Timer

logger = logging.getLogger(__name__)


class FrameProcessor:
    classifier: Classifier = None
    detector: Detector = None
    classified_plants_new: List[Plant] = None
    last_frame: np.ndarray = None

    def __init__(self, run_settings: Settings, source: Source):
        self.debug_settings = run_settings
        self.classifier = run_settings.classifier
        self.detector = run_settings.detector
        self.source = source
        if self.debug_settings.write_video:
            
            self.video_writer = cv2.VideoWriter(
                f"{self.debug_settings.output_folder}/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (source.width(), source.height())
            )

        self.image_output_folder = self.debug_settings.frame_output_folder
        self.classified_plants_new = []
        self.frames_without_plants = []

    def send_notification(self, plant: Plant):
        notifier = self.debug_settings.notifier
        message = f"Plant {plant.id} needs water"
        notifier.notify(message)

    def is_same_plant(self, plant: Plant, frame_nr: int, label: str, bbox):
        if frame_nr - 1 not in plant.frame_to_bounding_box:
            return str(label) == str(plant.id)

        return str(label) == str(plant.id) or plant.frame_to_bounding_box[frame_nr - 1].overlaps(bbox)

    def process_frame(self, frame: np.array, frame_nr: int):
        detected_plants = None
        if self.detector:
            with Timer("Detection", logger.debug):
                detected_plants = self.detector.predict([frame])

        optical_flow = None

        # Classify the image using the classifier
        if detected_plants:
            for bbox, label in detected_plants:
                plant_frame = frame.copy()
                plant_frame = bbox.cut_frame(plant_frame)
                logger.debug(
                    f"Object detector detected label: {label} at location: {bbox} with confidence: {bbox.score}"
                )

                # Get Optical Flow tracker
                is_new_plant = True
                current_plant = None

                if self.classified_plants_new:
                    for plant in self.classified_plants_new:
                        if self.is_same_plant(plant, frame_nr, label, bbox):
                            plant.frame_to_bounding_box[frame_nr] = bbox
                            current_plant = plant
                            is_new_plant = False
                            break

                if is_new_plant:
                    optical_flow_calculator = SparseOpticalFlowCalculator(
                        self.debug_settings, label
                    )  # if self.debug_settings.camera_mode # is CameraMode.MOVING else self.optical_flow_calculator
                    current_plant = Plant(label, {}, True, None, None, optical_flow_calculator)
                    current_plant.frame_to_bounding_box[frame_nr] = bbox
                    self.classified_plants_new.append(current_plant)
                    if self.debug_settings.write_plant_images:
                        cv2.imwrite(
                            self.debug_settings.plant_output_folder + f"/{label}_frame-{frame_nr}.jpg",
                            cv2.cvtColor(plant_frame, cv2.COLOR_RGB2BGR),
                        )

                # Calculate Optical Flow
                with Timer("Optical Flow"):
                    green_mask = None
                    if self.last_frame is not None:
                        if self.debug_settings.camera_mode is CameraMode.MOVING:
                            optical_flow = current_plant.optical_flow_calculator.calculate(plant_frame)
                        elif self.debug_settings.camera_mode is CameraMode.STATIC:
                            optical_flow = current_plant.optical_flow_calculator.calculate(frame, bbox)
                            needs_water = current_plant.optical_flow_calculator.classify(optical_flow)
                            if needs_water:
                                current_plant.is_healthy = False
                                current_plant.unhealthy_frames.append(frame_nr)
                                self.send_notification(current_plant)
                                if (
                                    self.debug_settings.write_plant_images
                                    and current_plant.unhealthy_frames[0] == frame_nr
                                ):
                                    filename = f"{self.debug_settings.plant_output_folder}/{label}_frame-{frame_nr}_needswater.jpg"
                                    cv2.imwrite(filename, cv2.cvtColor(plant_frame, cv2.COLOR_RGB2BGR))

                with Timer("Classification", logger.debug):
                    classifier_label = self.classifier.predict(plant_frame) if self.classifier else "No classifier"
                    # gc.collect()
                    logger.debug(f"Classifier detected label: {classifier_label}")

                if self.debug_settings.show_plant_frames:
                    cv2.putText(
                        plant_frame,
                        f"{label}, {classifier_label}, {bbox.score}",
                        (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(f"plant_frame {label}", cv2.cvtColor(plant_frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                if self.debug_settings.show_optical_flow and optical_flow is not None:
                    output_frame = frame.copy()
                    optical_arrows = current_plant.optical_flow_calculator.visualize(output_frame)
                    cv2.imshow("optical_flow_arrows", cv2.cvtColor(optical_arrows, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                if self.debug_settings.plot_optical_flow and optical_flow is not None:
                    if frame_nr % 10 == 0:
                        current_plant.optical_flow_calculator.plot(id=current_plant.id, save=True)

        else:
            logger.debug(f"Object detector did not detect any plants in frame {frame_nr}")
            self.frames_without_plants.append(frame_nr)

        output_frame = frame.copy()
        if self.debug_settings.show_bboxes:
            for plant in self.classified_plants_new:
                if frame_nr in plant.frame_to_bounding_box:
                    box_label(
                        output_frame, plant.frame_to_bounding_box[frame_nr], f"{plant.id}, {plant.classifier_label}"
                    )
        else:
            logger.debug(f"No bounding boxes to show in frame {frame_nr}")

        output_frame = cv2.putText(
            output_frame, f"Frame {frame_nr}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
        # Displaying the image if the debug setting is set
        if self.debug_settings.show_frame:
            cv2.imshow("frame", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.debug_settings.write_video:
            self.video_writer.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

        if self.debug_settings.write_frames:
            output_filename = os.path.join(self.image_output_folder, f"frame_{frame_nr}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

        self.last_frame = frame
