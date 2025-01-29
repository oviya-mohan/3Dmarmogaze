import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

import stereo_3D_box
import stereo_3D_face
import gaze_cone

# folders to use 
left_folder = "23102024/left_box/"
right_folder = "23102024/right_box/"

def get_box_coords(folder):

    # Load stereo calibration parameters (example)
    calibration_data = np.load( str(folder) + 'stereo_calibration_parameters.npz')
    K1 = calibration_data['mtxL']
    D1 = calibration_data['distL']
    K2 = calibration_data['mtxR']
    D2 = calibration_data['distR']
    R = calibration_data['R']
    T = calibration_data['T']

    # trasnform roation and translation matrices to real world coordinates
    R_wc = np.linalg.inv(R) # Equivalent to np.linalg.inv(rotation_matrix) if rotation_matrix is orthogonal
    t_wc = -np.dot(R_wc, T.reshape((3, 1)))

    # create extrinsic parameters?
    # calculate parameters for 3D triangulation based on one camera
    P1 = np.hstack((K1, np.zeros((3,1))))
    RT = np.hstack((R_wc,t_wc))
    K2_RT = np.dot(K2, RT)
    P2 = np.dot(K2,RT)

    ######################### define and triangulate box coordinates ####################

    # define coordinates for box from both left and right cameras (can be ontained via get_box_coords)

    if folder == "23102024/left_box/" :
         # 102324 - Left
        corners_right = np.array([[1067.0, 970.0, 184.0, 9.0, 681.0, 681.0, 253.0, 171.0], 
                                [374.0, 1236.0, 1382.0, 388.0, 472.0, 925.0, 985.0, 488.0]]) 
        corners_left = np.array([[1016.0, 886.0, 160.0, 9.0, 857.0, 814.0, 425.0, 392.0], 
                                [405.0, 1333.0, 1248.0, 468.0, 526.0, 982.0, 964.0, 543.0]])
    else:
        # 102324 - Right
        corners_right = np.array([[994.0, 847.0, 100.0, 9.0, 911.0, 843.0, 426.0, 420.0], 
                                  [437.0, 1410.0, 1252.0, 458.0, 541.0, 1026.0, 983.0, 540.0]])
        corners_left = np.array([[1070.0, 971.0, 179.0, 38.0, 705.0, 684.0, 273.0, 217.0], 
                                 [598.0, 1401.0, 1530.0, 608.0, 636.0, 1063.0, 1109.0, 641.0]])
       

    # 3D triangulate points for box and plot 
    points_4D = cv2.triangulatePoints(P2,P1, corners_left, corners_right)
    points_3D = points_4D[:3] / points_4D[3]
    points_3D = np.dot(R_wc, points_3D) + t_wc

    x_3d,y_3d,z_3d = points_3D
    # print("x_3d:", x_3d, "\n", "y_3d:", y_3d, "\n", "z_3d:", z_3d)    

    desired_x = [10, 10, 0, 0, 10, 10, 0, 0]
    desired_y = [10, 0, 0, 10, 10, 0, 0, 10]
    desired_z = [0, 0, 0, 0, 11.7, 11.7, 11.7, 11.7] 

    original_cube_points = np.array([x_3d, y_3d,z_3d]).T
    desired_cube_points = np.array([desired_x, desired_y, desired_z]).T

    # Compute the translation vector
    # Translate the first original corner to the new origin (0, 0, 0)
    translation_vector = -original_cube_points[2]  # Subtract the coordinates of the third corner
    translated_original_points = original_cube_points + translation_vector

    # Compute the rotation matrix
    # Use Procrustes analysis or least-squares fitting to compute the rotation
    H = np.dot(translated_original_points.T, desired_cube_points)  # Covariance matrix
    U, _, Vt = np.linalg.svd(H)  # Singular Value Decomposition
    rotation_matrix = np.dot(Vt.T, U.T)  # Optimal rotation matrix

    # Ensure the rotation matrix is proper (determinant = 1, avoiding reflection)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)

    # Apply the rotation to the translated points
    rotated_cube_points = np.dot(translated_original_points, rotation_matrix)
    print(rotated_cube_points)

    rotated_cube_x, rotated_cube_y, rotated_cube_z = rotated_cube_points.T

    origin = [x_3d[2], y_3d[2], z_3d[2]]
    x_3d = x_3d - origin[0]
    y_3d = y_3d - origin[1]
    z_3d = z_3d - origin[2]
    if folder == "23102024/right_box/":
        x_3d = x_3d + 14
    

    return x_3d, y_3d, z_3d, origin
    # return rotated_cube_x, rotated_cube_y, rotated_cube_z, origin
    # return desired_x, desired_y, desired_z, origin
            

def read_face_labels(folder) :

    ######################### load and triangulate face coordinates ####################

    # read labels from two csv files, left and right camera

    df_right = pd.read_csv( str(folder) +'left_labels.csv')
    # columns_to_remove = df_right.columns[df_right.iloc[2] == 'likelihood']
    columns_to_remove = df_right.columns[3::3]  
    df_right = df_right.drop(columns=columns_to_remove)
    # # print(df_right.head())
    # print(df_right.iloc[2].values)

    right_points = [[]]
    for index, row in df_right.iterrows():
        right_points.append(row[1:].to_numpy())  # Convert each row to an array
    right_points = right_points[3:]

    right_camera_x_coordinate = []
    right_camera_y_coordinate = []
    # print(right_points)
    for point in right_points:
        right_camera_x_coordinate.append(point[::2])
        right_camera_y_coordinate.append(point[1::2])

    df_left = pd.read_csv(str(folder) +'right_labels.csv')
    # columns_to_remove = df_left.columns[df_left.iloc[2] == 'likelihood']
    columns_to_remove = df_left.columns[3::3]  
    df_left = df_left.drop(columns=columns_to_remove)
    # print(df_left.head())
    # print(df_left.iloc[3])

    left_points = []
    for index, row in df_left.iterrows():
        left_points.append(row[1:].to_numpy())  # Convert each row to an array
    left_points = left_points[3:]

    left_camera_x_coordinate = []
    left_camera_y_coordinate = []
    # print(right_points)
    for point in right_points:
        left_camera_x_coordinate.append(point[::2])
        left_camera_y_coordinate.append(point[1::2])

    return right_camera_x_coordinate, right_camera_y_coordinate, left_camera_x_coordinate, left_camera_y_coordinate

def triangulate_face(folder, right_camera_x_coordinate, right_camera_y_coordinate, left_camera_x_coordinate, left_camera_y_coordinate, origin) :

    # Load stereo calibration parameters (example)
    calibration_data = np.load( str(folder) + 'stereo_calibration_parameters.npz')
    K1 = calibration_data['mtxL']
    D1 = calibration_data['distL']
    K2 = calibration_data['mtxR']
    D2 = calibration_data['distR']
    R = calibration_data['R']
    T = calibration_data['T']

    # trasnform roation and translation matrices to real world coordinates
    R_wc = np.linalg.inv(R) # Equivalent to np.linalg.inv(rotation_matrix) if rotation_matrix is orthogonal
    t_wc = -np.dot(R_wc, T.reshape((3, 1)))

    # create extrinsic parameters?
    # calculate parameters for 3D triangulation based on one camera
    P1 = np.hstack((K1, np.zeros((3,1))))
    RT = np.hstack((R_wc,t_wc))
    K2_RT = np.dot(K2, RT)
    P2 = np.dot(K2,RT)

    face_left = np.array([left_camera_x_coordinate.reshape(-1), left_camera_y_coordinate.reshape(-1)])
    face_left = np.array(face_left, dtype=np.float32)
        
    face_right = np.array([right_camera_x_coordinate, right_camera_y_coordinate])
    face_right = np.array(face_right, dtype=np.float32)

    facepoints_4D = cv2.triangulatePoints(P2,P1, face_left, face_right)
    facepoints_3D = facepoints_4D[:3] / facepoints_4D[3]
    facepoints_3D = np.dot(R_wc, facepoints_3D) + t_wc

    facex_3d, facey_3d, facez_3d = facepoints_3D 

    facex_3d = facex_3d - origin[0]
    facey_3d = facey_3d - origin[1]
    facez_3d = facez_3d - origin[2]

    if folder == "23102024/right_box/":
        facex_3d = facex_3d + 14

    return facex_3d, facey_3d, facez_3d
##################################################################################### 

######################### loop through frames and plot 3D ####################


# for frame in range(len(right_camera_x_coordinate)):
L_right_x, L_right_y, L_left_x, L_left_y = read_face_labels(left_folder)
R_right_x, R_right_y, R_left_x, R_left_y = read_face_labels(right_folder)

for frame in range(1): #for debugging

    fig = go.Figure()
    
    L_x_3d, L_y_3d, L_z_3d,left_origin = get_box_coords(left_folder)
    print(L_x_3d, L_y_3d, L_z_3d)
    stereo_3D_box.plot_box(fig, L_x_3d, L_y_3d, L_z_3d) # put this outside the loop to see movement through time
    # R_x_3d, R_y_3d, R_z_3d, right_origin = get_box_coords(right_folder)
    # stereo_3D_box.plot_box(fig, R_x_3d, R_y_3d, R_z_3d)

    # L_facex_3d, L_facey_3d, L_facez_3d = triangulate_face(left_folder, L_right_x[frame], L_right_y[frame], L_left_x[frame], L_left_y[frame], left_origin)
    # stereo_3D_face.plot_face(fig,L_facex_3d, L_facey_3d, L_facez_3d)
    # gaze_cone.draw_gaze_cone(fig,L_facex_3d, L_facey_3d, L_facez_3d)
    
    # R_facex_3d, R_facey_3d, R_facez_3d = triangulate_face(right_folder, R_right_x[frame], R_right_y[frame], R_left_x[frame], R_left_y[frame], right_origin)
    # stereo_3D_face.plot_face(fig,R_facex_3d, R_facey_3d, R_facez_3d)
    # gaze_cone.draw_gaze_cone(fig,R_facex_3d, R_facey_3d, R_facez_3d)



 # Update layout
# fig.update_layout(
#         scene=dict(
#                 xaxis_title='X axis',
#                 yaxis_title='Y axis',
#                 zaxis_title='Z axis'
#         ),
#         title='Interactive 3D Plot'
#         )
        # change visual orientation
    # fig.update_layout(
    #         scene=dict(
    #         camera=dict(
    #             # eye=dict(x=0.5, y=0.35, z=-1.5), # right_box
    #             eye=dict(x=-0.25, y=0.25, z=-1.75), # left_box
    #             up=dict(x=0, y=-1, z=0)  # This sets the up direction
    #         )))
    
    fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-12, 12], title='X Axis', showgrid=True, zeroline=True),
        yaxis=dict(range=[-12, 12], title='Y Axis', showgrid=True, zeroline=True),
        zaxis=dict(range=[-12, 12], title='Z Axis', showgrid=True, zeroline=True),
        camera=dict(
            eye=dict(x=0, y=0, z=-2),  # Camera positioned along the Y-axis looking at the X-Y plane
            up=dict(x=1, y=0, z=0)    # X-axis points "upward" on the screen
        )
    )
)

    fig.update_layout(showlegend=False)

    # Show the plot
    fig.show()
    # fig.write_image('3D_plots/frame_' + str(frame) + '.png', format='png')


