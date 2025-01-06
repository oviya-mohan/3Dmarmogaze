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
	calibration_videos/  (two calibration videos at a time for now, may add possibility to do all pairs at once in the future) <br>
	calibration_frames/ <br>
	stereo_calibration.py <br>
	stereo_calibration_parameters.npz (output) 
	
	
