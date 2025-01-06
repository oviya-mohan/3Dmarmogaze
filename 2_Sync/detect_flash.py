from skimage import io
import numpy as np
import os 
import glob
from os import listdir
from matplotlib import pyplot as plt
import cv2
 
 # calculate mean pixel value for the first 1000 frames 
 # (the flash should be within this range, for 60fps video 1000 frames is ~15 seconds)
frames_to_check = 1000

# number of frames after flash frame to begin trimmed video from 
# (for 60 fps video, 400 frames is ~7 seconds)
frames_to_skip = 400

#loop through all .MP4 files in the Raw folder
for mp4_file in glob.glob(os.path.join("Raw", '*.MP4')): 
    video = os.path.basename(mp4_file)  # Extract the video name without the path

    # open video
    cap = cv2.VideoCapture(mp4_file)
    frame_count = 0
    
    # intialize empty list to add mean pixel values of each frame 
    mean_pixel_values = []

    while cap.isOpened() and frame_count < frames_to_check:
        ret, frame = cap.read()
        
        # increment frame count
        frame_count += 1

        # append mean pixel value of current frame to list
        mean_pixel_values.append(np.mean(frame))

    # calculate the overall mean and standard deviation 
    # of the mean pixel value of the frames in the video 
    overall_mean = np.mean(mean_pixel_values)
    overall_sd = np.std(mean_pixel_values)
    print(len(mean_pixel_values), overall_mean, overall_sd)

    # loop through the frames again and find the frames 
    # whose mean pixel values are more than 2.5SD away 
    # from the overall mean of all the frames 
    # and save last frame with flash

    # reset frame count 
    for i in range(len(mean_pixel_values)):
        if mean_pixel_values[i] > overall_mean + 1.5*overall_sd:
            flash_frame = i 

    print(flash_frame)


    # get video properties to write output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))                    # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # Frame width
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # Total frame count

    # Set up the video writer for the trimmed output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter("Synced/" + str(video), fourcc, fps, (frame_width, frame_height))

    # Set the starting frame position to fixed number of frames frames from the last flash frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, flash_frame + frames_to_skip)

    # Loop through frames from the starting frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Write the frame to the output video
        out.write(frame)

    # Release video objects
    cap.release()
    out.release()
