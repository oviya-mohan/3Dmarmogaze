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

def calc_cone(start_point, end_point, angle):
    # Convert the angle to radians
    angle_rad = np.radians(angle)
    
    # Get the direction vector from start to end point
    direction = np.array(end_point) - np.array(start_point)
    
    # Normalize the direction vector to get the cone's axis
    direction = direction / np.linalg.norm(direction)
    
    # Calculate the distance between start and end points (length of the cone)
    length = np.linalg.norm(np.array(end_point) - np.array(start_point))
    
    # Calculate the radius of the cone's base (using tan(angle) * distance)
    radius = np.tan(angle_rad) * length
    
    # Create a circle (base of the cone) perpendicular to the direction vector
    num_points = 100  # Number of points on the circle
    theta = np.linspace(0, 2 * np.pi, num_points)
    
    # Perpendicular vector to the direction vector
    # Use the cross product to find a vector perpendicular to direction
    up = np.array([0, 0, 1])  # Default up vector
    if np.abs(direction[2]) > 0.9:  # Special case for vertical direction vector
        up = np.array([1, 0, 0])
    
    perpendicular_vector = np.cross(direction, up)
    perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
    
    # Points on the circle in the plane perpendicular to the direction vector
    circle_x = radius * np.cos(theta) * perpendicular_vector[0]
    circle_y = radius * np.cos(theta) * perpendicular_vector[1]
    circle_z = radius * np.cos(theta) * perpendicular_vector[2]
    
    # Now calculate the cone's surface points by scaling the circle points
    # from start point along the direction vector
    cone_x = start_point[0] + direction[0] * length + circle_x
    cone_y = start_point[1] + direction[1] * length + circle_y
    cone_z = start_point[2] + direction[2] * length + circle_z

    return cone_x, cone_y, cone_z
    

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

        # scale size of normal vector for visibility
        normal_length = 2
        normal_vector = normal_vector / np.linalg.norm(normal_vector) * normal_length

        # calculate the starting point of the normal vector on the plane midway between the eyes
        start_point = (left_eye + right_eye) / 2

        # define the end point of the normal vector
        end_point = start_point + normal_vector 

        angle = 10
        cone_x, cone_y, cone_z = calc_cone(start_point, end_point, angle)

    # draw normal vector on plot 
    fig.add_trace(go.Scatter3d(x=[start_point[0], end_point[0]], y=[start_point[1], end_point[1]],
                                z=[start_point[2], end_point[2]], mode='lines',
                                line=dict(color='black', width=5),
                                marker=dict(size=5, color='green'),
                                name='Normal Vector'))
    
    # draw the cone mesh
    fig = go.Figure(data=[go.Surface(
        x=cone_x, y=cone_y, z=cone_z, colorscale='Viridis', opacity=0.6)
    ])

