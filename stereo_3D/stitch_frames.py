import cv2
import os
import re

def stitch_images_to_video(folder_path, output_path, fps=1):
    # Get a list of all .png files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    
    # Sort files based on the frame number in the filename
    def extract_frame_number(file_name):
        match = re.search(r'frame_(\d+)\.png', file_name)
        return int(match.group(1)) if match else float('inf')

    image_files.sort(key=extract_frame_number)
    
    # Read the first image to get frame size
    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape
    
    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add images to video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_path}")

# Example usage
stitch_images_to_video(folder_path="23102024/right_box/combined_images", output_path="23102024/right_box/3D_video.mp4", fps=1)
