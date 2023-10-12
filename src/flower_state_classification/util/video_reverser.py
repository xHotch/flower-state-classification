import cv2
import os

"""
Script to reverse a video file.
"""

def reverse_video(input_file, output_file):

    video = cv2.VideoCapture(input_file)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the reversed video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # Read frames and write them in reverse order
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)

    for frame in reversed(frames):
        out.write(frame)

    # Release video objects
    video.release()
    out.release()

    print("Reversed video saved to", output_file)


if __name__ == "__main__":

    folder = r"C:\dev\videos\cut"
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            input_file_path = os.path.join(folder, file)
            export_folder_path = os.path.join(folder, f"reversed_{file}")
            reverse_video(input_file_path, export_folder_path)
