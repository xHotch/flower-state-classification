import cv2
import os

def reverse_video(input_file, output_file):
    # Open the video file
    video = cv2.VideoCapture(input_file)

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the reversed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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


def export_frames(input_file, output_folder, num_frames):
    # Open the video file
    video = cv2.VideoCapture(input_file)

    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(frame_count / num_frames)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read and export frames
    count = 0
    frame_number = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Export frame if it meets the step criteria
        if frame_number % step == 0:
            output_path = os.path.join(output_folder, f"frame{count}.jpg")
            cv2.imwrite(output_path, frame)
            count += 1

        frame_number += 1
        if count >= num_frames:
            break

    # Release video object
    video.release()

    print(f"{count} frames exported to {output_folder}")





if __name__ == "__main__":
    num_frames_to_export = 100

    folder = r"C:\dev\videos\cut"
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            input_file_path = os.path.join(folder, file)
            export_folder_path = os.path.join(folder, f"exported_{file}")
            export_frames(input_file_path, export_folder_path, num_frames_to_export)            