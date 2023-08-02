from enum import Enum, auto
import logging
import os
import time

from flower_state_classification.cv.models.ultralytics_models.ultralytics_models import UltralyticsDetector
from flower_state_classification.notifications.websocket_notifications import WebsocketNotifier
from flower_state_classification.notifications.websocket_server import WebsocketServer

# from flower_state_classification.notifications.telegram_notifications import TelegramBot

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    SCHEDULED = auto()
    CONTINUOUS = auto()


class CameraMode(Enum):
    STATIC = auto()
    MOVING = auto()


class Settings:
    # Classification Settings

    minimum_angle = 70  # Downwards angle is at 90 degrees
    maximum_angle = 110

    magnitude_threshold = 20  # Absolute size in pixel
    magnitude_threshold_scaled = 0.05  # Percentage of the plant height

    decay_factor = 0.9  # How much the magnitude decreases each frame

    # Source Settings
    camera_mode = CameraMode.STATIC

    # Scheduling Settings
    pipeline_mode = PipelineMode.CONTINUOUS
    daily_start_time = "10:00"
    daily_end_time = "18:00"
    every_x_minutes = 30

    ## Telegram Notification Settings
    telegram_token = ""  # Read from file or environment variable
    telegram_chat_id = ""  # Read from file or environment variable
    # Websocket Notification Settings
    websocket_host = "localhost"
    websocket_port = 8765
    # Notification Settings
    notifier = WebsocketNotifier(websocket_host, websocket_port)
    server = WebsocketServer(websocket_host, websocket_port)

    # Debug Output Settings
    show_frame = False
    show_bboxes = False
    write_video = False
    write_frames = True
    write_plant_images = True
    show_plant_frames = False

    ## Optical Flow Debug Settings
    show_optical_flow = False
    show_green_mask = False
    plot_optical_flow = True

    output_folder = r"./output"

    # Processing Settings
    detector = UltralyticsDetector("yolov8_m_openimages_best.pt", 0.3)
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
