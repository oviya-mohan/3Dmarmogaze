import cv2
import os
import numpy as np
import glob

# Parameters for checkerboard pattern
checkerboard_size = (8, 6)  # Number of inner corners per a chessboard row and column
square_size = 1.0  # Square size in your defined unit (e.g., meters or millimeters)


# extract frames from calibration video for stereo calibration for both cameras
frame_interval = 20  # Process every xth frame (adjust as needed)

calib_videos = [file for file in os.listdir("calibration_videos") if (file.endswith(".mp4") or file.endswith(".MP4")) ]
print(calib_videos)

# create and track output folders containing extracted frames
output_folders = []

for video in calib_videos:
    output_folder = "calibration_frames/" + str(video)
    output_folders.append(output_folder)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_path = "calibration_videos/" + str(video)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened() and saved_frame_count < 40:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                frame_filename = os.path.join(output_folder, f'frame_{saved_frame_count:04d}.png')
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved {saved_frame_count} frames containing the checkerboard pattern for " + str(video))



# Termination criteria for corner sub-pixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, e.g., (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
objpoints = []
imgpoints_left = []
imgpoints_right = []

# Read image pairs
images_right = glob.glob(output_folders[0] + str("/*.png"))
images_left = glob.glob(output_folders[1] + str("/*.png"))


for img_left, img_right in zip(images_left, images_right):
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    retL, cornersL = cv2.findChessboardCorners(grayL, checkerboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, checkerboard_size, None)

    # If found, add object points, image points (after refining them)
    if retL and retR:
        objpoints.append(objp)

        corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners2L)

        corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners2R)

        # Draw and display the corners
        cv2.drawChessboardCorners(imgL, checkerboard_size, corners2L, retL)
        cv2.drawChessboardCorners(imgR, checkerboard_size, corners2R, retR)
        cv2.imshow('Left', imgL)
        cv2.imshow('Right', imgR)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate each camera individually
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right,
                                                                mtxL, distL, mtxR, distR, grayL.shape[::-1],
                                                                criteria_stereo, flags)

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T)

# Save calibration results
np.savez('stereo_calibration_parameters.npz', ret=ret, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi1=roi1, roi2=roi2)

print("Stereo calibration completed successfully.")
