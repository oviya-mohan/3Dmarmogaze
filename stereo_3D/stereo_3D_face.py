import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def plot_face(fig, facex_3d, facey_3d, facez_3d):

    #color coding the face points 
    face_colors = ['purple', 'blue', 'cyan', 'green', 'orange', 'red']

    # plot face points 
    fig.add_trace(go.Scatter3d(
    x=facex_3d,
    y=facey_3d,
    z=facez_3d,
    mode='markers',
    marker=dict(size=5, color= face_colors),
    name='3D Points'
    ))

    ###################### Connect face points ###########################

    fig.add_trace(go.Scatter3d(
            x=[facex_3d[1], facex_3d[4]],
            y=[facey_3d[1], facey_3d[4]],
            z=[facez_3d[1], facez_3d[4]],
            mode='lines',
            line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter3d(
            x=[facex_3d[2], facex_3d[4]],
            y=[facey_3d[2], facey_3d[4]],
            z=[facez_3d[2], facez_3d[4]],
            mode='lines',
            line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter3d(
            x=[facex_3d[1], facex_3d[3]],
            y=[facey_3d[1], facey_3d[3]],
            z=[facez_3d[1], facez_3d[3]],
            mode='lines',
            line=dict(color='blue', width=2)
    ))
    

    for i in range(5):
            fig.add_trace(go.Scatter3d(
            x=[facex_3d[i], facex_3d[i + 1]],
            y=[facey_3d[i], facey_3d[i + 1]],
            z=[facez_3d[i], facez_3d[i + 1]],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Skeleton'
            ))

    fig.add_trace(go.Scatter3d(
            x=[facex_3d[2], facex_3d[3]],
            y=[facey_3d[2], facey_3d[3]],
            z=[facez_3d[2], facez_3d[3]],
            mode='lines',
            line=dict(color='blue', width=2)
    ))
    ######################################################################
