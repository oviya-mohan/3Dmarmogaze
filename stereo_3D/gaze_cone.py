import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

def calc_normal(A, B, C, P1, P2):
    # A - left_eye, B - right_eye, C - forehead, P1 - left_tuft, P2- right_tuft
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    P1 = np.array(P1)
    P2 = np.array(P2)

    # two vectors in the face plane
    AC = A - C
    BC = B - C
    
    # normal vector to the plane via cross product
    normal = np.cross (AC, BC)
    
    # vectors from C to P1 and C to P2
    CP1 = P1 - C
    CP2 = P2 - C
    
    # compute dot products to check the direction
    dot_P1 = np.dot(normal, CP1)
    dot_P2 = np.dot(normal, CP2)
    
    # both dot products being negative indicate the normal is in the right direction
    # if not, direction of normal needs to be reversed  
    if not (dot_P1 < 0 and dot_P2 < 0):
        normal = -normal  
    
    return normal

def calc_cone(start, direction, angle, height, resolution=100):
    # Normalize direction vector
    direction = direction / np.linalg.norm(direction)
    
    # Calculate radius of the cone base
    radius = height * np.tan(np.radians(angle))
    
    # Generate circular base points in 3D
    theta = np.linspace(0, 2 * np.pi, resolution)
    x_base = radius * np.cos(theta)
    y_base = radius * np.sin(theta)
    z_base = np.zeros_like(theta)
    
    # Rotate base to align with direction vector
    # Reference axis is Z
    ref = np.array([0, 0, 1])
    rot_axis = np.cross(ref, direction)
    rot_angle = np.arccos(np.dot(ref, direction))
    
    if np.linalg.norm(rot_axis) > 0:  # Avoid division by zero
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        K = np.array([
            [0, -rot_axis[2], rot_axis[1]],
            [rot_axis[2], 0, -rot_axis[0]],
            [-rot_axis[1], rot_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(rot_angle) * K + (1 - np.cos(rot_angle)) * (K @ K)
        base_points = np.vstack((x_base, y_base, z_base))
        rotated_base = R @ base_points
        x_base, y_base, z_base = rotated_base
    
    # Translate base to the tip of the cone
    x_base += start[0] + direction[0] * height
    y_base += start[1] + direction[1] * height
    z_base += start[2] + direction[2] * height
    
    # Add the tip of the cone
    x_cone = np.concatenate([[start[0]], x_base])
    y_cone = np.concatenate([[start[1]], y_base])
    z_cone = np.concatenate([[start[2]], z_base])
    
    return x_cone, y_cone, z_cone

    

def draw_gaze_cone(fig, facex_3d, facey_3d, facez_3d):
    # defining points for face plane and normal dir calculation
    left_eye = np.array([facex_3d[1], facey_3d[1], facez_3d[1]]) 
    right_eye = np.array([facex_3d[4], facey_3d[4], facez_3d[4]]) 
    forehead = np.array([facex_3d[2], facey_3d[2], facez_3d[2]])
    left_tuft = np.array([facex_3d[0], facey_3d[0], facez_3d[0]])
    right_tuft = np.array([facex_3d[5], facey_3d[5], facez_3d[5]])

    # calculate normal to the face plane defined by the eyes and forehead using above function

    # only works if 5/6 face points (except nose) are triangulated

    if (np.isnan(left_eye).any() or np.isnan(right_eye).any() or np.isnan(forehead).any() or np.isnan(left_tuft).any() or np.isnan(right_tuft).any()) == 0:

        normal_vector = calc_normal(left_eye, right_eye, forehead, left_tuft, right_tuft) 

        # factor to scale size of normal vector for visibility
        normal_length = 1 # 1 for unit normal vector

        # calculate the starting point of the normal vector on the plane midway between the eyes
        start_point = (left_eye + right_eye) / 2
        
        # define angle for cone (10 visual degrees) and caluclate cone parameters 
        angle = 10
        x_cone, y_cone, z_cone = calc_cone(start_point, normal_vector, angle, normal_length)

        # scale normal vector
        normal_vector = normal_vector / np.linalg.norm(normal_vector) * normal_length

        # define end point
        end_point = start_point + normal_vector

        # add normal vector on plot 
        fig.add_trace(go.Scatter3d(x=[start_point[0], end_point[0]], y=[start_point[1], end_point[1]],
                                    z=[start_point[2], end_point[2]], mode='lines',
                                    line=dict(color='black', width=5),
                                    marker=dict(size=5, color='green'),
                                    name='Normal Vector'))

        # triangulate the cone surface
        simplices = []
        for i in range(1, len(x_cone) - 1):
            simplices.append([0, i, i + 1])
        simplices.append([0, len(x_cone) - 1, 1])  # Close the base

        # create the 3D mesh for the cone and add to plot
        fig.add_trace(go.Mesh3d(
            x=x_cone,
            y=y_cone,
            z=z_cone,
            i=[s[0] for s in simplices],
            j=[s[1] for s in simplices],
            k=[s[2] for s in simplices],
            color='cyan',
            opacity=0.6
        ))
        

