# 3Dmarmogaze

Pipeline for 3D marmoset gaze tracking <br>
1. Set-up + Camera Config <br>
2. Sync videos via flash frames and trim <br>
	Raw <br>
	Synced <br>
	detect_flash.py - reads all .mp4 files in Raw and finds flash frames based on mean pixel values in the first xx frames and trims videos to begin yy frames from last flash frame and saves in Synced (xx - frames_to_check and yy  - frames_to_skip can be adjusted) 
	
