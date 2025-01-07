import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def plot_box(fig, x_3d,y_3d,z_3d):

        ###################### Connect box corners ###########################

        for i in range(0,3):
                fig.add_trace(go.Scatter3d(
                x=[x_3d[i], x_3d[i + 1]],
                y=[y_3d[i], y_3d[i + 1]],
                z=[z_3d[i], z_3d[i + 1]],
                mode='lines',
                line=dict(color='black', width=2),
                name='Box'
                ))

        for i in range(4,7):
                fig.add_trace(go.Scatter3d(
                x=[x_3d[i], x_3d[i + 1]],
                y=[y_3d[i], y_3d[i + 1]],
                z=[z_3d[i], z_3d[i + 1]],
                mode='lines',
                line=dict(color='black', width=2)
                ))        

        for i in range(0,4):
                fig.add_trace(go.Scatter3d(
                x=[x_3d[i], x_3d[i + 4]],
                y=[y_3d[i], y_3d[i + 4]],
                z=[z_3d[i], z_3d[i + 4]],
                mode='lines',
                line=dict(color='black', width=2)
                ))
                
        fig.add_trace(go.Scatter3d(
                x=[x_3d[0], x_3d[3]],
                y=[y_3d[0], y_3d[3]],
                z=[z_3d[0], z_3d[3]],
                mode='lines',
                line=dict(color='black', width=2)
                ))

        fig.add_trace(go.Scatter3d(
                x=[x_3d[4], x_3d[7]],
                y=[y_3d[4], y_3d[7]],
                z=[z_3d[4], z_3d[7]],
                mode='lines',
                line=dict(color='black', width=2)
                ))



