import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# https://docs.opencv.org/4.7.0/d4/dee/tutorial_optical_flow.html
class SparseOpticalFlowCalculator():

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict( maxCorners = 100,
    qualityLevel = 0.3,
    minDistance = 7,
    blockSize = 7 )

    # Lucas Kanade parameters
    lk_params = dict( winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def __init__(self) -> None:
        self.color = np.random.randint(0, 255, (100, 3))
        pass
    
    def preprocess_frame(self, frame: np.array) -> np.array:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame_gray
    
    def calculate_optical_flow(self, frame1: np.array, frame2: np.array) -> np.array:
        """
        Calculates the optical flow between two frames.
        """
        output = np.zeros_like(frame1)

        # Convert to grayscale
        frame1_gray = self.preprocess_frame(frame1)
        frame2_gray = self.preprocess_frame(frame2)

        # Find corners in first frame
        p0 = cv2.goodFeaturesToTrack(frame1_gray, mask = None, **self.feature_params)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Create a mask image for drawing purposes
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            output = cv2.circle(output, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            output = cv2.add(output, mask)

        return output
    
class DenseOpticalFlowCalculator():

    """
    Calculates the dense optical flow between two frames.
    """
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0
    last_frame: np.array = None
    
    def calculate(self, new_frame: np.ndarray) -> np.ndarray:
        if self.last_frame is None:
            logger.debug("First frame for optical flow")
            self.last_frame = new_frame
            return None
        calculated_flow = self.calculate_optical_flow(self.last_frame, new_frame)
        return calculated_flow

    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Calculates the optical flow between two frames.
        """
        # Handle different input sizes
        if frame1.shape != frame2.shape:
            logger.warning(f"Frames have different shapes: {frame1.shape} and {frame2.shape}. Padding with zeros.")

            if frame1.shape[0] < frame2.shape[0]:
                frame2 = frame2[0:frame1.shape[0], :,:]
            if frame1.shape[1] < frame2.shape[1]:
                frame2 = frame2[:, 0:frame1.shape[1],:]

            if frame1.shape != frame2.shape:
                frame2 = cv2.copyMakeBorder(frame2, 0, frame1.shape[0] - frame2.shape[0], 0, frame1.shape[1] - frame2.shape[1], cv2.BORDER_CONSTANT, value=[0,0,0])

        # Convert to grayscale
        try:
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        except:
            logger.warning(f"Could not convert frames to grayscale: {frame1.shape} and {frame2.shape}.")
            return output
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, self.pyr_scale, self.levels, self.winsize, self.iterations, self.poly_n, self.poly_sigma, self.flags)

        # Convert to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Convert to BGR
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return output