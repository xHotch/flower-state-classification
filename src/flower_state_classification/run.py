import argparse
import os
import time
import cv2
import numpy as np
from flower_state_classification.cli import add_parsers

from flower_state_classification.debug.debugsettings import DebugSettings
from flower_state_classification.cv.frame_processor import FrameProcessor
from flower_state_classification.input.filesource import VideoFileSource
from flower_state_classification.input.imagefoldersource import ImageFolderSource
from flower_state_classification.input.webcamsource import WebcamSource
from flower_state_classification.util.benchmark import benchmark_fps, benchmark_time
from flower_state_classification.util.summary import create_summary

import logging

logger=logging.getLogger(__name__)

class FlowerStateClassificationPipeline:
    def __init__(self, filename, debug_settings):
        if filename:
            if os.path.isdir(filename):
                self.source = ImageFolderSource(filename)
            else:
                self.source = VideoFileSource(filename)
        else:
            self.source = WebcamSource()
        self.debug_settings = debug_settings
        self.frame_processor = FrameProcessor(debug_settings, self.source)

    @benchmark_time
    def run(self):
        logger.info(f"Starting pipeline on {self.source}")
        self.frame_number = 0
        while True:
            if not self.run_loop():
                break
            self.frame_number += 1
        logger.info(f"Pipeline finished on {self.source}")

    @benchmark_fps(cooldown = 1)
    def run_loop(self):
        hasframe, frame = self.source.get_frame()
        if not hasframe:
            return False
        self.frame_processor.process_frame(frame, self.frame_number)
        return True


def main(source=None):
    pipeline = FlowerStateClassificationPipeline(source, DebugSettings())
    logging.basicConfig(level=DebugSettings.log_level)
    pipeline.run()
    create_summary(pipeline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parsers(parser)
    main(**vars(parser.parse_args()))
