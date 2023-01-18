from dataclasses import dataclass


@dataclass
class InputData:
    frame_count: int
    resolution: tuple  # (width, height)
    video_path: str = None  # path to video file, if None, use webcam
