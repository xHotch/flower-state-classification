from abc import ABC, abstractmethod
import copy
import os
from typing import List, Tuple
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['animation.html'] = 'jshtml'
import numpy as np
import logging

from flower_state_classification.data.boundingbox import BoundingBox
from flower_state_classification.debug.settings import Settings

logger = logging.getLogger(__name__)

class OpticalFlowCalculator(ABC):
    # Color parameters
    filter_green: bool = False
    green_threshold_low: np.array = np.array([36, 25, 25])
    green_threshold_high: np.array = np.array([90, 255, 255]) 

    # Debug variables
    flow_angles = []
    flow_magnitudes = []

    total_angles = []
    total_magnitudes = []

    needs_water = False
    previously_notified = False

    def __init__(self, debug_settings: Settings) -> None:
        self.debug_settings = debug_settings
        self.fig, (self.axs, self.axs_mag) = plt.subplots(2, sharex=True)
        self.fig.suptitle("Optical flow over time")

        # self.fig.show()

    def get_green_mask(self, frame: np.ndarray, bbox: BoundingBox = None) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.green_threshold_low, self.green_threshold_high)
        if bbox:
            bbox.mask_frame(mask, value = 0)
        
        if self.debug_settings.show_green_mask:
            if self.debug_settings.show_green_mask and mask is not None:
                cv2.imshow("green_mask", mask)
                cv2.waitKey(1)

        return mask
    

    @abstractmethod
    def calculate(self, frame: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def visualize(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def classify(self, optical_flow: np.ndarray) -> None:
        ...

    def _calculate_angle_and_magnitude(self, optical_flow: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        if optical_flow is None:
            return
        
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
            return

        average_angle = np.mean(angle)
        median_angle = np.median(angle)

        average_magnitude = np.mean(magnitude)
        median_magnitude = np.median(magnitude)

        logger.info("Optical flow angle: average: {}, median: {}".format(average_angle, median_angle))
        logger.info("Optical flow magnitude: average: {}, median: {}".format(average_magnitude, median_magnitude))

        return angle, magnitude
    
    def plot(self, angles, magnitudes, id: str = None, save = False):
        
        average_angle = [np.mean(angle) for angle in angles]
        average_magnitude = [np.mean(magnitude) for magnitude in magnitudes]

        #axs_mag.xlabel("Frame number [int]")
        #axs.ylabel("Angle [degrees]")
        self.axs.plot(average_angle)

        #axs_mag.ylabel("Magnitude [px]")
        self.axs_mag.plot(average_magnitude)
        
        minimum_angle = self.debug_settings.minimum_angle
        maximum_angle = self.debug_settings.maximum_angle

        self.axs.axhline(y=minimum_angle, color='g', linestyle='--', label="Minimum angle", linewidth=2)
        self.axs.axhline(y=maximum_angle, color='g', linestyle='--', label="Maximum angle", linewidth=2)
        

        if self.needs_water and not self.previously_notified:
            self.axs.axvline(x=len(angles), color='r', linestyle='--')
            self.axs_mag.axvline(x=len(angles), color='r', linestyle='--')
            self.previously_notified = True

        if save:
            folder = self.debug_settings.get_output_folder("optical_flow")
            self.fig.savefig(os.path.join(folder,f"{id}.png"))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# https://docs.opencv.org/4.7.0/d4/dee/tutorial_optical_flow.html
class SparseOpticalFlowCalculator(OpticalFlowCalculator):
    
    last_points: List[Tuple] = []

    def __init__(self, settings = None) -> None:
        super().__init__(settings)
        self.color = np.random.randint(0, 255, (100, 3))
        ...

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict( maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7 )

    last_frame: np.ndarray = None

    # Lucas Kanade parameters
    lk_params = dict( winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def calculate(self, new_frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        frame = copy.deepcopy(new_frame)
        if self.last_frame is None:
            logger.debug("First frame for optical flow")
            self.last_frame = frame
            self.last_bbox = bbox
            return None
        calculated_flow = self.calculate_optical_flow(self.last_frame, frame, self.last_bbox, bbox)
        self.last_frame = frame
        self.last_bbox = bbox
        return calculated_flow
    
    def preprocess_frame(self, frame: np.array) -> np.array:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame_gray
    
    def visualize(self, image: np.ndarray, old_keypoints, new_keypoints) -> np.ndarray:
        # Create a mask image for drawing purposes
        output = np.zeros_like(image)
        frame = image.copy()
        im2 = image.copy()
        for i, (new, old) in enumerate(zip(new_keypoints, old_keypoints)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            im2 = cv2.arrowedLine(im2, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            output = cv2.line(output, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            output = cv2.add(frame, output)

        return im2
    
    def calculate_optical_flow(self, last_frame: np.array, current_frame: np.array, last_bbox: BoundingBox = None, current_bbox: BoundingBox = None) -> np.array:
        """
        Calculates the optical flow between two frames.
        """
        # Convert to grayscale
        frame1_gray = self.preprocess_frame(last_frame)
        frame2_gray = self.preprocess_frame(current_frame)

        # Find corners in first frame
        mask = self.get_green_mask(last_frame, last_bbox)

        if not self.last_points:
            p0 = cv2.goodFeaturesToTrack(frame1_gray, mask = mask, **self.feature_params)
        else:
            p0 = self.last_points[-1][0]
            # p0 = cv2.goodFeaturesToTrack(frame1_gray, mask = mask, **self.feature_params)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        flow = np.array(good_new - good_old)

        self.last_points.append((p1,st))
        return flow

    def classify(self, optical_flow: np.ndarray) -> bool:
        angle, magnitude = self._calculate_angle_and_magnitude(optical_flow, None)
        self.flow_angles.append(angle)
        self.flow_magnitudes.append(magnitude)
        
        if len(self.last_points) < 2:
            return False
        total_flow = self.last_points[-1][0] - self.last_points[0][0]
        total_angle, total_magnitude = self._calculate_angle_and_magnitude(total_flow, None)
        self.total_angles.append(total_angle)
        self.total_magnitudes.append(total_magnitude)

        total_angle = np.mean(total_angle)
        total_magnitude = np.mean(total_magnitude)

        if total_angle > self.debug_settings.minimum_angle and total_angle < self.debug_settings.maximum_angle:
            logger.info("Optical flow classified as downwards")

            if total_magnitude > self.debug_settings.magnitued_threshold:
                logger.info("Optical flow classified as downwards and strong")
                #TODO notifiy
                self.needs_water = True
                return True
            else:
                logger.info("Optical flow classified as downwards but not enough magnitude")

        return False

class DenseOpticalFlowCalculator(OpticalFlowCalculator):
    """
    Calculates the dense optical flow between two frames. Only takes into account the green pixels.
    """
    # Optical flow parameters
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    last_frame: np.ndarray = None

    def __init__(self, settings = None) -> None:
        super().__init__(settings)

    def calculate(self, new_frame: np.ndarray, bbox: BoundingBox = None) -> np.ndarray:
        frame = copy.deepcopy(new_frame)
        if self.last_frame is None:
            logger.debug("First frame for optical flow")
            self.last_frame = frame
            self.last_bbox = bbox
            return None
        
        calculated_flow = self.calculate_optical_flow(self.last_frame, frame)
        self.last_frame = frame
        self.last_bbox = bbox
        return calculated_flow

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if self.filter_green:
            mask = self.get_green_mask(frame)
            frame = cv2.bitwise_and(frame, frame, mask=mask)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame
    
    def visualize(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        image = image.copy()

        # Convert image to RGB (if it's in BGR format)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate the magnitude and angle of the optical flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Create a mask to only display strong optical flow vectors
        mask = np.zeros_like(image)
        mask[..., 1] = 255
        valid_flow = magnitude > 1
        mask[valid_flow] = [0, 0, 255]

        # Calculate the start and end points of the optical flow vectors
        start_points = np.meshgrid(range(0, image.shape[1], 10), range(0, image.shape[0], 10))
        start_points = np.array(start_points, dtype=np.float32).T.reshape(-1, 2)
        end_points = start_points + flow[start_points[:, 1].astype(int), start_points[:, 0].astype(int)]

        # Draw arrows on the image
        for pt1, pt2 in zip(start_points, end_points):
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            cv2.arrowedLine(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA, 0, 0.4)

        return image

    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculates the optical flow between two frames.
        """
        output = np.zeros_like(frame1)
        # Handle different input sizes
        if frame1.shape != frame2.shape:
            logger.warning(f"Frames have different shapes: {frame1.shape} and {frame2.shape}. Padding with zeros.")

            if frame1.shape[0] < frame2.shape[0]:
                frame2 = frame2[0:frame1.shape[0], :,:]
            if frame1.shape[1] < frame2.shape[1]:
                frame2 = frame2[:, 0:frame1.shape[1],:]

            if frame1.shape != frame2.shape:
                frame2 = cv2.copyMakeBorder(frame2, 0, frame1.shape[0] - frame2.shape[0], 0, frame1.shape[1] - frame2.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])

        # Preprocess frames
        try:
            frame1_gray = self._preprocess(frame1)
            frame2_gray = self._preprocess(frame2)

        except:
            logger.warning(f"Could not preprocess frames")
            return output
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, self.pyr_scale, self.levels, self.winsize, self.iterations, self.poly_n, self.poly_sigma, self.flags)

        return flow
    
    def convert_optical_flow_to_polar(self, optical_flow: np.ndarray, normalize:bool = False) -> np.ndarray:
        if optical_flow is None:
            return None
        output = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        output[..., 1] = 255
        output[..., 0] = ang * 180 / np.pi / 2
        if normalize:
            output[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        else:
            output[..., 2] = mag
        output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
        return output

    def classify(self, optical_flow: np.ndarray) -> None:
        self._calculate_angle_and_magnitude(optical_flow, self.get_green_mask(self.last_frame, self.last_bbox))