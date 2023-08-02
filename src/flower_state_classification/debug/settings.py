from enum import Enum, auto
import logging
import os
import time

from flower_state_classification.cv.models.ultralytics_models.ultralytics_models import UltralyticsDetector
from flower_state_classification.notifications.telegram_notifications import TelegramBot

logger = logging.getLogger(__name__)

class PipelineMode(Enum):
    SCHEDULED = auto()
    CONTINUOUS = auto()

class CameraMode(Enum):
    STATIC = auto()
    MOVING = auto()

class Settings:
    # Classification Settings

    minimum_angle = 80 # Downwards angle is at 90 degrees
    maximum_angle = 100

    magnitued_threshold = 30 # TODO Maybe make dependent on size of plant

    # Source Settings
    camera_mode = CameraMode.STATIC

    # Scheduling Settings
    pipeline_mode = PipelineMode.CONTINUOUS
    daily_start_time = "10:00"
    daily_end_time = "18:00"
    every_x_minutes = 30

    # Notification Settings
    notifier = TelegramBot
    ## Telegram Notification Settings
    telegram_token = "" # Read from file or environment variable

    # Debug Output Settings
    show_frame = True
    show_bboxes = True
    write_video = True
    write_images = True
    show_plant_frames = False

    ## Optical Flow Debug Settings
    show_optical_flow = True
    show_green_mask = True
    plot_optical_flow = True

    output_folder = r"./output"

    # Processing Settings
    detector = UltralyticsDetector()
    classifier = None
    
    # Logging Settings
    log_level = logging.INFO

    # Data Settings
    dataset_download_folder = r"C:\dev\datasets"

    # GPU Settings
    try:
        import torch
        if torch.cuda.is_available():
            device = "GPU"
            logger.info("Using GPU")
        else:
            device = "CPU"
            logger.info("Using CPU")
    except:
        device = "CPU"
        logger.info("Could not import torch, using CPU")

    def __init__(self) -> None:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        self.output_folder = os.path.join(self.output_folder, time_string)
        os.makedirs(self.output_folder, exist_ok=True)

        if self.write_images:
            self.image_output_folder = os.path.join(self.output_folder,"frames")
            os.makedirs(self.image_output_folder, exist_ok=True)

        self.log_file = os.path.join(self.output_folder, "log.txt")

    def get_output_folder(self, foldername):
        folder = os.path.join(self.output_folder, foldername)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    