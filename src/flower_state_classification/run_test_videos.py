import os
from flower_state_classification.input.videofilesource import VideoFileSource
from flower_state_classification.run import main as run_main

"""
Script to run the pipeline on all videos in a folder.

Can be used to test the pipeline on a large number of videos.
max_number_of_frames can be used to limit the number of (equally spaced) frames that are processed per video. 
"""

def main():
    video_folder = r"C:\dev\videos\cut"
    list_of_videos = [file for file in os.listdir(video_folder) if file.endswith(".mp4")]
    list_of_videos.sort(reverse=True)
    max_number_of_frames = [10, 20, 50, 100, 500, 1000]

    for video in list_of_videos:
        for max_frames in max_number_of_frames:
            try:
                source = VideoFileSource(os.path.join(video_folder, video), max_frames)
                run_main(source, video)
            except Exception as e:
                print(f"Error while processing {video}: {e}")
                print("-----------------------------------")


if __name__ == "__main__":
    main()
