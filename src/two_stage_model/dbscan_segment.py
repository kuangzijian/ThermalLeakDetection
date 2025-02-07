import cv2
import numpy as np
import os
from diff import calculate_ssim
import imageio
from net import ResNet, BasicBlock, Bottleneck
from collections import namedtuple
import torch
from api.utils import cv2_2_pil
import torchvision.transforms as transforms
import torch.nn.functional as F
import joblib
from sklearn.svm import SVC
from bin_svm import FeatureExtractor
from collections import defaultdict
from sklearn.cluster import DBSCAN


svm = joblib.load("svm.m")

def init_model(path='tut5-model.pt'):
    if path is None:
        return None
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet50_config, 2).to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

diff_means = [0.924, 0.924, 0.924]
diff_stds = [0.072, 0.072, 0.072]

color_means = [0.669, 0.342, 0.479]
color_stds= [0.215, 0.212, 0.132]

diff_transforms = transforms.Compose([
                           transforms.Resize(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = diff_means,
                                                std = diff_stds)
                       ])

color_transforms = transforms.Compose([
                           transforms.Resize(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = color_means,
                                                std = color_means)
                       ])

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    return blurred

def process_images_from_folder(folder_path, criterion, show=True, debug=False, use_model=True, use_svm=False):
    if "diff" in use_model:
        test_transforms = diff_transforms
    else:
        test_transforms = color_transforms

    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])

    model = init_model(use_model)
    feature_extractor = FeatureExtractor(model)
    
    prev_frame = None
    frames_for_gif = []
    detected_frames = 0
    undetected_frames = 0
    detections_through_frames = []
    
    for i, filename in enumerate(image_files):
        file_path = os.path.join(folder_path, filename)
        frame = cv2.imread(file_path)
        
        if frame is None:
            continue

        current_frame = preprocess_frame(frame)
        H, W = current_frame.shape

        detections_per_frame = 0
        
        if prev_frame is not None:
            if criterion == "abs":
                frame_delta = cv2.absdiff(prev_frame, current_frame)
                thresh = cv2.threshold(frame_delta, 10, 255, cv2.THRESH_BINARY)[1]
                # contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            elif criterion == "ssim":
                frame_delta = calculate_ssim(prev_frame, current_frame)
                frame_delta *= 255.0
                frame_delta = frame_delta.astype("uint8")
                thresh = cv2.threshold(frame_delta, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                thresh = cv2.erode(thresh, (5,5), iterations=3)
            elif criterion == "combo":
                frame_delta_abs = cv2.absdiff(prev_frame, current_frame)
                thresh_abs = cv2.threshold(frame_delta_abs, 10, 255, cv2.THRESH_BINARY)[1]
                frame_delta_win = calculate_ssim(cv2.blur(prev_frame, (3,3)), cv2.blur(current_frame, (3,3)))
                frame_delta_win *= 255.0
                frame_delta_win = frame_delta_win.astype("uint8")
                thresh_win = cv2.threshold(frame_delta_win, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                thresh_win = cv2.erode(thresh_win, (5,5), iterations=3)
                frame_delta_combo = (255 - frame_delta_abs) // 2 + frame_delta_win // 2
                # thresh = cv2.threshold(frame_delta_combo, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                thresh = cv2.threshold(255 - frame_delta_combo, 30, 255, cv2.THRESH_BINARY)[1]
                # thresh = cv2.adaptiveThreshold(frame_delta_combo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1)
                thresh = cv2.dilate(thresh, (5,5), iterations=3)
                thresh = cv2.erode(thresh, (5,5), iterations=3)
                cv2.imwrite(f"thresholds/{i}.png", thresh)

            cam_info_area = (30, 450)
            thresh[:cam_info_area[0], :cam_info_area[1]] = np.zeros(cam_info_area)
            if debug:
                cv2.imwrite("thresh.jpg", thresh)
                cv2.imwrite("diff.jpg", frame_delta)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            detections = 0
            if use_model:
                if "diff" in use_model: source = frame_delta_combo
                else: source = frame

            if not contours:
                continue
            centroids = np.array([np.mean(contour, axis=0)[0] for contour in contours])
            db = DBSCAN(eps=50, min_samples=1).fit(centroids)  # eps might need adjustment based on image resolution
            labels = db.labels_

            for label in np.unique(labels):
                if label == -1:
                    continue  # Skip noise
                
                # Get the centroids for the current cluster
                cluster_mask = (labels == label)
                cluster_points = centroids[cluster_mask]
                
                # Calculate the bounding rectangle for each cluster
                x, y, w, h = cv2.boundingRect(cluster_points.astype(np.int32))
                if 500 < w * h < 5e4:
                    newx, newy, xw, yh = x - 50, y- 50, x + w, y + h
                    if newy < 0: newy = 0
                    if yh+50 > H: yh = H
                    if newx < 0: newx = 0
                    if xw+50 > W: xw = W
                    cur_patch = source[newy:yh, newx:xw]
                    cur_patch = cv2_2_pil(cur_patch)
                    transformed_patch = test_transforms(cur_patch)
                    output = feature_extractor(transformed_patch.unsqueeze(0)).detach().numpy()
                    output = np.expand_dims(output.squeeze(), axis=0)
                    is_normal = svm.predict(output)

                    if is_normal:
                        continue
                    cv2.rectangle(frame, (newx, newy), (xw ,yh), (0, 255, 0), 2)
                detections += 1

            if detections:
                detected_frames += 1
            else:
                undetected_frames += 1

            cv2.putText(frame, f"Detected: {detected_frames} Undetected: {undetected_frames}", (int(W*0.45), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_for_gif.append(frame_rgb)

            if show:
                cv2.imshow('Detected Changes', frame)

                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
        prev_frame = current_frame
        detections_through_frames.append(detections_per_frame)
    if show:
        cv2.destroyAllWindows()
    return frames_for_gif, detections_through_frames

folder_path = '../dataset/LeakAI dataset 2023-12-20/618/618-50-01-0038-01/4/2023-12-20/negative'
criterion = "combo"

frames, _ = process_images_from_folder(folder_path, criterion, show=True, debug=False, use_model='tut5-model_diff.pt', use_svm=True)
