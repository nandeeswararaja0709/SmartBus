import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
  
  ret, frame = cap.read()
  if ret == True:
    # Define start and end points for the line
    start_point = (0, 250)
    end_point = (700, 250)
    
    # Define the color (BGR) and thickness of the line
    color = (0, 255, 0)  # Green color
    thickness = 2
    # load our serialized model from disk
    # cv2.putText(frame, "-Prediction border - Entrance-", 0,
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Draw the line on the frame
    frame = cv2.line(frame, start_point, end_point, color, thickness)
    
    # Display the resulting frame
    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()