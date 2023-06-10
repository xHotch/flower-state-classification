import logging
import os
import time

class DebugSettings:

    # Video Settings
    show_frame = True
    show_bboxes = True
    write_video = False
    write_images = True
    show_plant_frames = False

    # Logging Settings
    log_level = logging.DEBUG

    # Output Settings
    output_folder = r"E:\dev\flower-state-classification\output"

    # Data Settings
    dataset_download_folder = r"C:\dev\datasets"

    def __init__(self) -> None:
        time_string = time.strftime("%Y%m%d-%H%M%S")
        self.output_folder = os.path.join(self.output_folder, time_string)
        os.makedirs(self.output_folder, exist_ok=True)