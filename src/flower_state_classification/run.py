import argparse
import os
import cv2
from flower_state_classification.cli import add_parsers

from flower_state_classification.debug.debugsettings import DebugSettings
from flower_state_classification.cv.frame_processor import FrameProcessor
from flower_state_classification.input.filesource import VideoFileSource
from flower_state_classification.input.imagefoldersource import ImageFolderSource
from flower_state_classification.input.webcamsource import WebcamSource

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
        self.frame_processor = FrameProcessor(debug_settings)

    def run(self):
        logger.info(f"Starting pipeline on {self.source}")
        while True:
            hasframe, frame = self.source.get_frame()
            if not hasframe:
                break
            self.frame_processor.process_frame(frame)


def main(source=None):
    pipeline = FlowerStateClassificationPipeline(source, DebugSettings())
    logging.basicConfig(level=DebugSettings.log_level, force=True)
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parsers(parser)
    main(**vars(parser.parse_args()))
