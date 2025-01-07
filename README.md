# 3Dmarmogaze

Pipeline for 3D marmoset gaze tracking <br>
1. Set-up + Camera Config <br>
2. Sync videos via flash frames and trim <br>
	Raw/ <br>
	Synced/ <br>
	detect_flash.py - reads all .mp4 files in Raw and finds flash frames based on mean pixel values in the first xx frames and trims videos to begin yy frames from last flash frame and saves in Synced (xx - frames_to_check and yy  - frames_to_skip can be adjusted) <br>
3. DLC <br>
4. 3D Triangulation  <br>
	a. pair wise stereo calibration : takes synced calibration videos as input and clacualtes stereo calibration parameters for a pair of cameras <br>
	calibration_videos/  (two calibration videos - left and right - at a time for now, may add possibility to do all pairs at once in the future) <br>
	calibration_frames/ <br>
	stereo_calibration.py <br>
	stereo_calibration_parameters.npz (output) 
	b. pair wise 3D triangulation of box and face points <br> 
	left_labels.csv (labels for face points from left camera) <br> 
	right_labels.csv (labels for face points from right camera) <br> 
	stereo_calibration_parameters.npz (from 4a) <br>
	stereo_3D_box.py - code to plot box <br>
	stereo_3D_face.py - code to plot face points <br> 
	stereo_3D_plot.py - define box corner coordinates, read in face labels from csv, read in camera parameters, triangulate 3D coordinates for box corners and face points (using stereo_3D_box.py and stereo_3D_face.py) and plot everything together for each frame - currently only displays frame (plotly interactive frame) <br>
	Saved frames stitched into video: ![current example](https://drive.google.com/uc?export=view&id=1OCP2ramtKjVqt7a1vF60PJgLwIiJY74o)
	

	
	
	
	
