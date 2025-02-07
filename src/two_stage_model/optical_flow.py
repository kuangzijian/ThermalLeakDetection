import cv2
import numpy as np
import os
import glob

# Function to get good features to track
def get_good_features_to_track(gray_frame):
    feature_params = dict(maxCorners=300, qualityLevel=0.5, minDistance=5, blockSize=7)
    return cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)

# Directory containing the images
image_folder = '../dataset/LeakAI dataset 2023-12-20/618/618-50-01-0038-01/3/2023-12-20/positive'
image_sequence = sorted([f for f in glob.glob(image_folder + "/*.jpg")])



import cv2
import numpy as np

# Function to load images (replace with your actual method for loading images)
def load_images(image_sequence):
    frame_iterator = []
    for image_name in image_sequence:
        frame_iterator.append(cv2.imread(image_name)[30:])
    return frame_iterator


# Parameters for ShiTomasi corner detection (feature selection)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Load the first frame and find corners in it
frame_iterator = load_images(image_sequence)
def process_frame(frame, background):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Frame differencing
    diff = cv2.absdiff(background, gray_frame)
    
    # Thresholding to binarize
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

# Load the first frame as background (assuming a static background for simplification)
first_frame = frame_iterator[0]
background = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Assuming you have a method to iterate over frames
for frame in frame_iterator[1:]:
    contours = process_frame(frame, background)
    
    # Draw contours
    for contour in contours:
        if cv2.contourArea(contour) > 100: # Filter out small contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
    
    cv2.imshow('Liquid Detection', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()