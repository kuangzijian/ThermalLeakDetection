import cv2
import numpy as np


def region_growing(img, seed, threshold=10):
    """
    img: Original image
    seed: A tuple (x, y) representing the seed point
    threshold: The intensity difference threshold
    """
    
    # Initialize the region growing criteria
    intensity_difference = threshold
    dimensions = img.shape[:2]
    region_size = 1
    segmented_img = np.zeros(dimensions, np.uint8)
    
    # Create a list of pixels to be examined
    seed_list = [seed]
    
    # This is the value we're going to match the other pixel values to
    match_value = img[seed]

    while seed_list:
        # Pop a seed from the seed list
        current_pixel = seed_list.pop(0)
        x, y = current_pixel
        
        # Try the four neighbors (up, down, left, right) for region growing criteria
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < dimensions[0] and 0 <= new_y < dimensions[1]: # Stay within image boundaries
                # Check if the pixel is similar enough to the seed
                pixel_value = img[new_x, new_y]
                if np.all(abs(pixel_value - match_value) < intensity_difference):
                    # If it meets the criteria, add it to the segmented image
                    segmented_img[new_x, new_y] = 255
                    # Add the new pixel to the seed list to grow the region
                    seed_list.append((new_x, new_y))
                    # Update the match_value to the mean of the region
                    match_value = ((region_size * match_value) + pixel_value) / (region_size + 1)
                    region_size += 1

    return segmented_img

import cv2
import numpy as np
import os
from diff import calculate_ssim

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    return blurred

def process_images_from_folder(folder_path, criterion, show=True, debug=False, use_model=True, use_svm=False, use_cluster=False):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])
    
    prev_frame = None
    frames_for_gif = []
    detections_through_frames = []
    
    for i, filename in enumerate(image_files):
        file_path = os.path.join(folder_path, filename)
        frame = cv2.imread(file_path)
        
        if frame is None:
            continue

        current_frame = preprocess_frame(frame)
        H, W = current_frame.shape
        if prev_frame is not None:
            frame_delta_abs = cv2.absdiff(prev_frame, current_frame)
            frame_delta_win = calculate_ssim(cv2.blur(prev_frame, (3,3)), cv2.blur(current_frame, (3,3)))
            frame_delta_win *= 255.0
            frame_delta_win = frame_delta_win.astype("uint8")
            thresh_win = cv2.threshold(frame_delta_win, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh_win = cv2.erode(thresh_win, (5,5), iterations=3)
            frame_delta_combo = (255 - frame_delta_abs) // 2 + frame_delta_win // 2
            thresh = cv2.threshold(255 - frame_delta_combo, 30, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, (5,5), iterations=3)

            cam_info_area = (30, 450)
            thresh[:cam_info_area[0], :cam_info_area[1]] = np.zeros(cam_info_area)

            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]

            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    seed_point = (cX, cY)

                    # Grow regions from seed points on the original image
                    grown_region = region_growing(frame, seed_point, 10)
            
            if show:
                cv2.imshow('Detected Changes', grown_region)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        prev_frame = current_frame
        if filename == "61850010038013850010038013850010038013850010038013231220171019578593.jpg":
            cv2.imwrite("1.png", thresh)
    if show:
        cv2.destroyAllWindows()
    return frames_for_gif

folder_path = '../dataset/LeakAI dataset 2023-12-20/618/618-50-01-0038-01/3/2023-12-20/positive'
criterion = "combo"

frames = process_images_from_folder(folder_path, criterion, show=True, debug=False, use_model="tut5-model_4ch_v3.pt", use_svm=False, use_cluster=False)
