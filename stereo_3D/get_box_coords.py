import cv2
import os
import argparse

# marmo_videos = [file for file in os.listdir("marmo_videos") if (file.endswith(".mp4") or file.endswith(".MP4")) ]
# print(marmo_videos)

# Function to capture mouse clicks
def click_event(event, x, y, flags, params):
    global click_count  # Use the global click_count variable
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count < max_clicks:
            click_count += 1
            x_coords.append(float(x))
            y_coords.append(float(y))
            
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)  # Optional: Visualize click
            cv2.imshow("Image", image)
        else:
            print("X Coordinates: ", x_coords)
            print("Y Coordinates: ", y_coords, "\n")
            print("Press Esc. to exit")
            cv2.destroyAllWindows()  # Close the window after 8 clicks

def read_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None
    
    # Read the first frame
    success, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if success:
        # Convert the frame from BGR to RGB (OpenCV uses BGR format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        print("Error: Unable to read the first frame.")
        return None


# parser = argparse.ArgumentParser(description="Get box coordinates for a video")
    
# # Add argument for file name
# parser.add_argument("filename", type=str, help="name of the video file in .mp4")
    
# # Parse the arguments
# args = parser.parse_args()
    
# # Get the file name from the command line argument
# video = args.filename
# image = read_first_frame("marmo_videos/" + str(video))

image = cv2.imread("box/right_102324.jpg")
x_coords, y_coords = [], []

# Display image and set up mouse callback
# print("Get box coordinates for " + str(video))

click_count = 0
max_clicks = 8

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()