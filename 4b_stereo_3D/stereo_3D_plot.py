import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

import stereo_3D_box
import stereo_3D_face

######################### load and define calibration parameter ####################
# camera naming convention still Left and Right

# Load stereo calibration parameters (example)
calibration_data = np.load('stereo_calibration_parameters.npz')
K1 = calibration_data['mtxL']
D1 = calibration_data['distL']
K2 = calibration_data['mtxR']
D2 = calibration_data['distR']
R = calibration_data['R']
T = calibration_data['T']

# print(R)
# print(T)

# trasnform roation and translation matrices to real world coordinates
R_wc = np.linalg.inv(R) # Equivalent to np.linalg.inv(rotation_matrix) if rotation_matrix is orthogonal
t_wc = -np.dot(R_wc, T.reshape((3, 1)))

# create extrinsic parameters?
# calculate parameters for 3D triangulation based on one camera
P1 = np.hstack((K1, np.zeros((3,1))))
RT = np.hstack((R_wc,t_wc))
K2_RT = np.dot(K2, RT)
P2 = np.dot(K2,RT)

# print("World Transformed Parameters")
# print(R_wc)
# print(t_wc)

#####################################################################################

######################### define and triangulate box coordinates ####################

# define coordinates for box from both left and right cameras  (can be obtained via napari) 
#corners_left = np.array([[], []]) 
# example
corners_left = np.array([[1055.489428	,	978.7120111	,	172.5491323	,	14.47797956	,	689.6676175	,	682.8931396	,	262.8755052	,	206.4215221]	,[610.6143649	,	1414.519084	,	1547.750484	,	633.1959581,639.9704361,	1066.762548	,	1111.925735	,	649.0030734]])


# corners_left_undistort = cv2.undistortImagePoints(corners_left, K2, D2)

#corners_right = np.array([[], []])
# example
corners_right = np.array([[983.2283297	,	913.2253907	,	170.2909729	,	5.445342265	,	908.709072	,	883.8693194	,	468.3680038	,	420.9466579]	,[	472.8666461	,	1432.584359	,	1321.934552	,	569.967497	,	549.6440631	,	1014.824884	,	1010.308565	,	585.7746123]])


# corners_right_undistort = cv2.undistortImagePoints(corners_right, K1, D1)

# 3D triangulate points for box and plot 
points_4D = cv2.triangulatePoints(P2,P1, corners_left, corners_right)
points_3D = points_4D[:3] / points_4D[3]
points_3D = np.dot(R_wc, points_3D) + t_wc

x_3d,y_3d,z_3d = points_3D

#####################################################################################

######################### load and triangulate face coordinates ####################

# read labels from two csv files, left and right camera

df_right = pd.read_csv('right_labels.csv')
right_points = [[]]
for index, row in df_right.iterrows():
    right_points.append(row[3:].to_numpy())  # Convert each row to an array
right_points = right_points[3:]

right_camera_x_coordinate = []
right_camera_y_coordinate = []
# print(right_points)
for point in right_points:
    right_camera_x_coordinate.append(point[::2])
    right_camera_y_coordinate.append(point[1::2])

df_left = pd.read_csv('left_labels.csv')
left_points = []
for index, row in df_left.iterrows():
    left_points.append(row[3:].to_numpy())  # Convert each row to an array
left_points = left_points[3:]

left_camera_x_coordinate = []
left_camera_y_coordinate = []
# print(right_points)
for point in right_points:
    left_camera_x_coordinate.append(point[::2])
    left_camera_y_coordinate.append(point[1::2])

##################################################################################### 

######################### loop through frames and plot 3D ####################

fig = go.Figure()
stereo_3D_box.plot_box(fig, x_3d,y_3d,z_3d)

# for frame in range(len(right_camera_x_coordinate)):
for frame in range(1): #for debugging

    print(frame)

    face_left = np.array([left_camera_x_coordinate[frame].reshape(-1), left_camera_y_coordinate[frame].reshape(-1)])
    face_left = np.array(face_left, dtype=np.float32)

    print(face_left)
        
    face_right = np.array([right_camera_x_coordinate[frame], right_camera_y_coordinate[frame]])
    face_right = np.array(face_right, dtype=np.float32)

    print(face_right)

    facepoints_4D = cv2.triangulatePoints(P2,P1, face_left, face_right)
    facepoints_3D = facepoints_4D[:3] / facepoints_4D[3]
    facepoints_3D = np.dot(R_wc, facepoints_3D) + t_wc

    facex_3d, facey_3d, facez_3d = facepoints_3D 
    stereo_3D_face.plot_face(fig,facex_3d, facey_3d, facez_3d)


 # Update layout
fig.update_layout(
        scene=dict(
                xaxis_title='X axis',
                yaxis_title='Y axis',
                zaxis_title='Z axis'
        ),
        title='Interactive 3D Plot'
        )
        # change visual orientation
fig.update_layout(
        scene=dict(
        camera=dict(
        #     eye=dict(x=0.5, y=0.35, z=-1.5),
            eye=dict(x=0.5, y=0.35, z=-1.75),
            up=dict(x=0, y=-1, z=0)  # This sets the up direction
        )))

fig.update_layout(showlegend=False)

# Show the plot
fig.show()


