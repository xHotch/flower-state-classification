from enum import Enum, auto
import logging
import os
import time

from flower_state_classification.cv.models.ultralytics_models.ultralytics_models import UltralyticsDetector
from flower_state_classification.notifications.websocket_notifications import WebsocketNotifier
from flower_state_classification.notifications.websocket_server import WebsocketServer


logger = logging.getLogger(__name__)


class Settings:

    # Classification Settings
    '''
    Sets the minimum and maximum angle of the optical flow vectors to be considered as downward motion.
    
    (The vertical downwards angle is at 90 degrees).
    '''
    minimum_angle = 60  # degrees
    maximum_angle = 120 # degrees

    '''
    Sets the minimum magnitude threshold of the optical flow vectors to be considered strong enough motion.
    
    The magnitude_threshold is scaled by the magnitude_threshold_scaled times the bounding box size, to obtain the actual threshold.
    '''
    magnitude_threshold = 20  # Absolute size in pixel
    magnitude_threshold_scaled = 0.05  # Percentage of the plant height


    # Scheduling Settings
    '''
    Switches between using a scheduled webcam and a live webcam.

    Scheduled webcams will only be active between the daily_start_time and daily_end_time.
    Frames will be taken every cooldown_in_minutes minutes.
    '''
    use_scheduled_webcam = True
    daily_start_time = "10:00"
    daily_end_time = "17:00"
    cooldown_in_minutes = 1

    # Websocket Notification Settings
    '''
    The websocket host and port to use for the websocket server and notifier
    '''
    websocket_host = "localhost"
    websocket_port = 8765
    # Notification Settings
    notifier = WebsocketNotifier(websocket_host, websocket_port)
    server = WebsocketServer(websocket_host, websocket_port)

    # Debug Output Settings
    '''
    Different flags to enable and disable debug output.
    '''
    show_frame = True
    show_bboxes = True
    write_video = True
    write_frames = True
    write_plant_images = True
    show_plant_frames = True

    ## Optical Flow Debug Settings
    show_optical_flow = True
    show_green_mask = False
    plot_optical_flow = True

    # Output Settings
    output_folder = r"./output"

    # Processing Settings
    '''
    Setup the object detector and classifier to use.

    We experimented with using plant classifiers to help identify specific plants, however currently no classifier is supported.
    '''
    detector = UltralyticsDetector("yolov8_m_openimages_best.pt", 0.3)
    classifier = None

    # Logging Settings
    log_level = logging.INFO


    def __init__(self, folder=None) -> None:
        if folder:
            self.output_folder = os.path.join(self.output_folder, folder)
        time_string = time.strftime("%Y%m%d-%H%M%S")
        self.output_folder = os.path.join(self.output_folder, time_string)
        os.makedirs(self.output_folder, exist_ok=True)
        self.frame_output_folder = None
        if self.write_frames:
            self.frame_output_folder = os.path.join(self.output_folder, "frames")
            os.makedirs(self.frame_output_folder, exist_ok=True)

        if self.write_plant_images:
            self.plant_output_folder = os.path.join(self.output_folder, "plants")
            os.makedirs(self.plant_output_folder, exist_ok=True)

        self.log_file = os.path.join(self.output_folder, "log.txt")
        self.setup_logging()

    def setup_logging(self):
        loghandlers = [logging.StreamHandler(), logging.FileHandler(self.log_file)]
        logging.basicConfig(
            level=self.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=loghandlers
        )

    def get_output_folder(self, foldername):
        folder = os.path.join(self.output_folder, foldername)
        os.makedirs(folder, exist_ok=True)
        return folder
