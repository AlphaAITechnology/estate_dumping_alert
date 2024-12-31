import cv2
import numpy as np
import gzip as gz

# Open a video file or capture from a camera
cap = cv2.VideoCapture('./20240806_1803.mkv')

# Create the background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=50)


hc_mask = None
with gz.open("street_container_mask_581.csv.gz", 'rt') as file:
    hc_mask = np.loadtxt(file).astype(np.uint8)
with gz.open("street_backdrop_mask_581.csv.gz", 'rt') as sm_file:
    hc_mask = np.loadtxt(sm_file).astype(np.uint8) * hc_mask
hardcoded_mask = np.stack((hc_mask, hc_mask, hc_mask), axis=2)


cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
while True:
    # Read a new frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Display the original frame and the foreground mask
    fg_mask = np.where(fg_mask>0, np.ones_like(fg_mask), np.zeros_like(fg_mask))*255
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)), iterations=3)
    fg_mask_ = np.stack((fg_mask, fg_mask, fg_mask), axis=2)
    
    cv2.imshow('Frame', np.hstack((fg_mask_ * hardcoded_mask, frame * hardcoded_mask)))

    # Exit if the user presses the 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
