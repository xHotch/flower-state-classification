import argparse
import os
import time
import cv2
import numpy as np
from flower_state_classification.cli import add_parsers

from flower_state_classification.settings.settings import Settings
from flower_state_classification.cv.frame_processor import FrameProcessor
from flower_state_classification.input.scheduled_webcamsource import ScheduledWebcamsource
from flower_state_classification.input.source import Source
from flower_state_classification.input.videofilesource import VideoFileSource
from flower_state_classification.input.imagefoldersource import ImageFolderSource
from flower_state_classification.input.webcamsource import WebcamSource
from flower_state_classification.util.summary import create_summary

import logging

from flower_state_classification.util.timer import Timer

logger = logging.getLogger(__name__)


class FlowerStateClassificationPipeline:
    def __init__(self, source, run_settings: Settings):
        if issubclass(type(source), Source):
            self.source = source
        elif source:
            if os.path.isdir(source):
                self.source = ImageFolderSource(source)
            else:
                self.source = VideoFileSource(source)
        else:
            if run_settings.use_scheduled_webcam:
                self.source = ScheduledWebcamsource(run_settings.daily_start_time, run_settings.daily_end_time, run_settings.cooldown_in_minutes)
            else:
                self.source = WebcamSource()
        self.run_settings = run_settings
        self.frame_processor = FrameProcessor(run_settings, self.source)

    @Timer(name="Total Runtime", logger=logger.info)
    def run(self):
        logger.info(f"Starting pipeline on {self.source}")
        self.frame_number = 0
        try:
            while True:
                if not self.run_loop():
                    break
                self.frame_number += 1
        except KeyboardInterrupt:
            ...
        logger.info(f"Pipeline finished on {self.source}")
        cv2.destroyAllWindows()

    @Timer("Run Loop", logger=logger.debug)
    def run_loop(self):
        hasframe, frame = self.source.get_frame()
        if not hasframe:
            return False
        self.frame_processor.process_frame(frame, self.frame_number)
        return True


def main(source=None, output_folder=None):
    run_settings = Settings(output_folder)
    run_settings.setup_logging()
    pipeline = FlowerStateClassificationPipeline(source, run_settings)

    pipeline.run()
    create_summary(pipeline)

    Timer.print_summary(logging.info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parsers(parser)
    main(**vars(parser.parse_args()))
