from abc import ABC, abstractmethod
import copy
import os
from typing import Dict, List, Tuple
import cv2
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("agg")
plt.rcParams["animation.html"] = "jshtml"
import numpy as np
import logging

from flower_state_classification.data.boundingbox import BoundingBox
from flower_state_classification.settings.settings import Settings

logger = logging.getLogger(__name__)


class OpticalFlowCalculator(ABC):

    # Color parameters
    green_threshold_low: np.array
    green_threshold_high: np.array

    # Debug variables
    flow_angles: List[np.ndarray]
    flow_magnitudes: List[np.ndarray]

    total_angles: List[np.ndarray]
    total_magnitudes: List[np.ndarray]

    status: List
    previously_notified = False

    def __init__(self, debug_settings: Settings, plant_id="") -> None:
        logger.warning("Optical flow calculator initialized")
        self.flow_angles = []
        self.flow_magnitudes = []
        self.total_angles = []
        self.total_magnitudes = []
        self.debug_settings = debug_settings
        self.green_threshold_low = np.array(debug_settings.green_mask_lower)
        self.green_threshold_high = np.array(debug_settings.green_mask_upper)

        self.status = []
        self.plant_id = plant_id
        self.scaled_threshold = None

    def get_green_mask(self, frame: np.ndarray, bbox: BoundingBox = None) -> np.ndarray:
        """
        Filter out non green values from the frame. If a boundingbox is given,  removes the pixels outside the boundingbox.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.green_threshold_low, self.green_threshold_high)
        if bbox:
            mask = bbox.mask_frame(mask, value=0)

        if self.debug_settings.show_green_mask and mask is not None:
            cv2.imshow("green_mask", mask)
            cv2.waitKey(1)

        return mask

    @abstractmethod
    def calculate(self, frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Calculates the optical flow between two frames.
        """
        ...

    @abstractmethod
    def visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Returns an image that visualizes the current optical flow
        """
        ...

    @abstractmethod
    def classify(self, optical_flow: np.ndarray) -> None:
        """
        Calculate whether the optical flow indicates that the plant needs watering.
        """
        ...

    def _calculate_angle_and_magnitude(
        self, optical_flow: np.ndarray, mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if optical_flow is None:
            return None, None

        if mask is not None:
            optical_flow = cv2.bitwise_and(optical_flow, optical_flow, mask=mask)

        if len(optical_flow.shape) == 2 or optical_flow.shape[2] == 2:
            magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1], angleInDegrees=True)

        else:
            angle = optical_flow[..., 0] * 2
            magnitude = optical_flow[..., 2]

        angle = angle[angle != 0]
        magnitude = magnitude[magnitude != 0]

        if not angle.any() or not magnitude.any():
            logger.debug("No optical flow detected")
            return None, None

        average_angle = np.mean(angle)
        median_angle = np.median(angle)

        average_magnitude = np.mean(magnitude)
        median_magnitude = np.median(magnitude)

        logger.debug(
            "Plant: {}, Optical flow angle: average: {}, median: {}".format(self.plant_id, average_angle, median_angle)
        )
        logger.debug("Optical flow magnitude: average: {}, median: {}".format(average_magnitude, median_magnitude))

        return angle, magnitude

    def _plot_angle_and_magnitude(self, angles: np.ndarray, magnitudes: np.ndarray) -> None:
        average_angle = [np.mean(angle) for angle in angles]
        average_magnitude = [np.mean(magnitude) for magnitude in magnitudes]

        # axs_mag.xlabel("Frame number [int]")
        # axs.ylabel("Angle [degrees]")
        plot_angle = self.axs.plot(average_angle)

        # axs_mag.ylabel("Magnitude [px]")
        plot_magnitude = self.axs_mag.plot(average_magnitude)

        return plot_angle, plot_magnitude

    def plot(self, id: str = None, save=False):
        """
        Plots the optical flow over time. If save is set to true, the plot is saved to the output folder.
        To display the plot in a live window, the backend matplotlib is using might need to be changed (see line 9 matplotlib.use("agg"))
        """
        if self.flow_angles is None or len(self.flow_angles) < 1:
            logger.warning("No optical flow to plot")
            return
        
        self.fig, (self.axs, self.axs_mag) = plt.subplots(2, sharex=True)
        self.fig.suptitle("Optical flow over time")
        self.axs.set_title("Angle")
        self.axs.set_ylabel("Angle [degrees]")
        self.axs_mag.set_title("Magnitude")
        self.axs_mag.set_ylabel("Magnitude [px]")
        self.axs_mag.set_xlabel("Frame number [scalar]")
        plot_flow_angle, plot_flow_magnitude = self._plot_angle_and_magnitude(self.flow_angles, self.flow_magnitudes)
        plot_total_angle, plot_total_magnitude = self._plot_angle_and_magnitude(self.total_angles, self.total_magnitudes)

        minimum_angle = self.debug_settings.minimum_angle
        maximum_angle = self.debug_settings.maximum_angle

        line_min_angle = self.axs.axhline(y=minimum_angle, color="g", linestyle="dotted", label="Minimum angle", linewidth=0.5)
        line_max_angle = self.axs.axhline(y=maximum_angle, color="r", linestyle="dotted", label="Maximum angle", linewidth=0.5)

        line_mag = self.axs_mag.axhline(
            y=self.debug_settings.magnitude_threshold,
            color="g",
            linestyle="--",
            label="Magnitude threshold",
            linewidth=0.5,
        )
        line_mag_scaled = self.axs_mag.axhline(y=self.scaled_threshold, color="r", linestyle="--", label="Scaled threshold", linewidth=0.5)

        area_needs_water = None
        start_idx = None
        for index, status in self.status:
            if status == "needs_watering" and start_idx is None:
                start_idx = index
            elif not status == "needs_watering" and start_idx is not None:
                end_idx = index

                area_needs_water = self.axs.axvspan(start_idx, end_idx, alpha=0.2, color="blue", label="Needs watering")
                self.axs_mag.axvspan(start_idx, end_idx, alpha=0.2, color="blue")

                # self.axs.fill_between(x_values[start_idx:end_idx], 0, 1, alpha=0.3, color='blue')
                # self.axs_mag.fill_between(x_values[start_idx:end_idx], 0, 1, alpha=0.3, color='blue')

                start_idx = None

        if start_idx is not None and self.status[-1][1] == "needs_watering":
            end_idx = len(self.status)
            area_needs_water = self.axs.axvspan(start_idx, end_idx, alpha=0.2, color="blue", label="Needs watering")
            self.axs_mag.axvspan(start_idx, end_idx, alpha=0.2, color="blue")


        if area_needs_water is not None:
            self.fig.legend(handles = [plot_flow_angle[0], plot_total_angle[0], line_min_angle, line_max_angle, line_mag, line_mag_scaled, area_needs_water], labels = ["Average", "Accumulated", "Minimum angle", "Maximum angle", "Magnitude threshold", "Scaled threshold", "Needs watering"], loc="center right", bbox_to_anchor=(1.3, 0.6))
        else:
            self.fig.legend(handles = [plot_flow_angle[0], plot_total_angle[0], line_min_angle, line_max_angle, line_mag, line_mag_scaled], labels = ["Average", "Accumulated", "Minimum angle", "Maximum angle", "Magnitude threshold", "Scaled threshold"], loc="center right", bbox_to_anchor=(1.3, 0.6))

        plt.tight_layout()
        if save:
            folder = self.debug_settings.get_output_folder("optical_flow")
            self.fig.savefig(os.path.join(folder, f"{id}.png"), dpi=500, bbox_inches="tight")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()
        plt.close(self.fig)


# Mostly taken from  https://docs.opencv.org/4.7.0/d4/dee/tutorial_optical_flow.html
class SparseOpticalFlowCalculator(OpticalFlowCalculator):
    """
    Use the Sparse Lucas-Kanade method to calculate the optical flow.
    """
    last_points: List[Tuple]

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    last_frame: np.ndarray = None

    # Lucas Kanade parameters
    lk_params = dict(
        winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    def __init__(self, settings=None, plant_id="") -> None:
        super().__init__(settings, plant_id)
        self.color = np.random.randint(0, 255, (100, 3))
        self.last_points = []

    def calculate(self, new_frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        frame = copy.deepcopy(new_frame)
        if self.last_frame is None:
            logger.debug("First frame for optical flow")
            self.last_frame = frame
            self.last_bbox = bbox
            return None
        calculated_flow = self._calculate_optical_flow(self.last_frame, frame, self.last_bbox, bbox)
        if calculated_flow is not None:
            self.last_frame = frame
            self.last_bbox = bbox
        return calculated_flow

    def _preprocess_frame(self, frame: np.array) -> np.array:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame_gray

    def visualize(self, image: np.ndarray) -> np.ndarray:
        old_keypoints, new_keypoints = self.last_points[0][0], self.last_points[-1][0]

        # Create a mask image for drawing purposes
        output = np.zeros_like(image)
        frame = image.copy()
        im2 = image.copy()
        for i, (new, old) in enumerate(zip(new_keypoints, old_keypoints)):
            a, b = old.ravel()
            c, d = new.ravel()

            im2 = cv2.arrowedLine(im2, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            output = cv2.line(output, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            output = cv2.add(frame, output)

        return im2

    def _calculate_optical_flow(
        self,
        last_frame: np.array,
        current_frame: np.array,
        last_bbox: BoundingBox = None,
        current_bbox: BoundingBox = None,
    ) -> np.array:
        """
        Calculates the optical flow between two frames.
        """
        # Convert to grayscale
        frame1_gray = self._preprocess_frame(last_frame)
        frame2_gray = self._preprocess_frame(current_frame)

        # Find corners in first frame
        mask = self.get_green_mask(last_frame, last_bbox)

        if not self.last_points:
            p0 = cv2.goodFeaturesToTrack(frame1_gray, mask=mask, **self.feature_params)
            if p0 is None:
                logger.warning("No corners found for plant {}".format(self.plant_id))
                return None
            logger.warning("Found {} corners for plant {}".format(len(p0), self.plant_id))
        else:
            p0 = self.last_points[-1][0]
            # p0 = cv2.goodFeaturesToTrack(frame1_gray, mask = mask, **self.feature_params)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        flow = np.array(good_new - good_old)

        self.last_points.append((p1, st))
        return flow

    def classify(self, optical_flow: np.ndarray) -> bool:
        angle, magnitude = self._calculate_angle_and_magnitude(optical_flow, None)

        if angle is None or magnitude is None:
            logger.warning("Could not calculate angle or magnitude for plant {}".format(self.plant_id))
            return False

        self.flow_angles.append(angle)
        self.flow_magnitudes.append(magnitude)

        index = len(self.flow_angles)
        if len(self.last_points) < 2:
            self.status.append((index, "enough_water"))
            return False
        total_flow = self.last_points[-1][0] - self.last_points[0][0]
        total_angle, total_magnitude = self._calculate_angle_and_magnitude(total_flow, None)

        self.total_angles.append(total_angle)
        self.total_magnitudes.append(total_magnitude)

        total_angle = np.mean(total_angle)
        total_magnitude = np.mean(total_magnitude)

        self.scaled_threshold = (
            self.debug_settings.magnitude_threshold_scaled * self.last_bbox.height()
            if not self.scaled_threshold
            else self.scaled_threshold
        )

        if total_angle > self.debug_settings.minimum_angle and total_angle < self.debug_settings.maximum_angle:
            logger.debug("Optical flow classified as downwards")

            if total_magnitude > self.scaled_threshold:
                logger.debug("Optical flow classified as downwards and strong")

                self.status.append((index, "needs_watering"))
                return True
            else:
                logger.debug("Optical flow classified as downwards but not enough magnitude")

        self.status.append((index, "enough_water"))
        return False