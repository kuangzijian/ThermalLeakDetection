import logging
import cv2
import numpy as np
import os
from two_stage_model.diff import calculate_ssim
from two_stage_model.net import ResNet, Bottleneck
from collections import namedtuple
import torch
from two_stage_model.api.utils import cv2_2_pil
import torchvision.transforms as transforms
import torch.nn.functional as F
from collections import defaultdict
from two_stage_model.utils import check_date_folder, check_label_folder, is_final_path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LeakingBoundingBox:
    def __init__(self, class_label: str=None, xc: float=None, yc: float=None, 
                 width: float=None, height: float=None, confidence: float=None, attr_str: str = None):
        """
        Initializes a bounding box with the specified attributes.
        
        Args:
            class_label (str): The class label of the object.
            xc (float): The x-coordinate of the center of the bounding box (YOLO format).
            yc (float): The y-coordinate of the center of the bounding box (YOLO format).
            width (float): The width of the bounding box (YOLO format).
            height (float): The height of the bounding box (YOLO format).
            confidence (float): The confidence score of the detection.
        """
        if attr_str and any([class_label, xc, yc, width, height, confidence]):
            raise ValueError("Please either pass a YOLO format string or individual attributes")
        if attr_str:
            self.__from_str(attr_str)
        else:
            self.class_label = class_label
            self.coords = [xc, yc, width, height]
            self.attrs = [class_label, *self.coords, confidence]
            self.confidence = confidence

    def __repr__(self):
        return (f"LeakingBoundingBox(class_label='{self.class_label}', coordinates={tuple(self.coords)}, confidence={self.confidence})")
    
    def __getitem__(self, key):
        return self.attrs[key]

    def __from_str(attr_str: str):
        parts = attr_str.strip().split()
        if len(parts) != 6:
            raise ValueError("A YOLO formatting string must have 6 elements")
        return LeakingBoundingBox(
            class_label=parts[0],
            xc=float(parts[1]),
            yc=float(parts[2]),
            width=float(parts[3]),
            height=float(parts[4]),
            confidence=float(parts[5])
        )
    
class PotentialLeaking:
    def __init__(self, x: float=None, y: float=None, 
                 xw: float=None, yh: float=None, ancestors: List[int]=None):
        self.coords = [x, y, xw, yh]
        self.ancestors = ancestors

def init_ResNetmodel(path=None, mc=False):
    if path is None:
        return None
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet50_config, 2).to(device)
    if mc:
        model.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

diff_means = [0.924, 0.924, 0.924]
diff_stds = [0.072, 0.072, 0.072]

color_means = [0.669, 0.342, 0.479]
color_stds= [0.215, 0.212, 0.132]

means = [0.669, 0.342, 0.479, 0.924]
stds= [0.215, 0.212, 0.132, 0.072]

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

mc_transforms = transforms.Compose([
                           transforms.Resize(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means,
                                                std = stds)
                       ])

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    return blurred

def boxes_overlap(box1, box2):
    x1, y1, x1_w, y1_h = box1
    x2, y2, x2_w, y2_h = box2
    return not (x1_w < x2 or x2_w < x1 or y1_h < y2 or y2_h < y1)

def find_clusters(bounding_boxes, contours):
    adj_list = defaultdict(list)
    n = len(bounding_boxes)
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(bounding_boxes[i], bounding_boxes[j]):
                adj_list[i].append(j)
                adj_list[j].append(i)

    visited = set()
    clusters = []
    cluster_sources = []

    def dfs(node, cluster, source):
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                cluster.append(bounding_boxes[neighbor])
                source.append(contours[neighbor])
                dfs(neighbor, cluster, source)

    for i in range(n):
        if i not in visited:
            visited.add(i)
            cluster = [bounding_boxes[i]]
            source = [contours[i]]
            dfs(i, cluster, source)
            clusters.append(cluster)
            cluster_sources.append(source)
    
    return clusters, cluster_sources

def trim(box, x_bound, y_bound):
    min_x, min_y, max_x, max_y = box
    if min_y < 0: min_y = 0
    if max_y > y_bound: max_y = y_bound
    if min_x < 0: min_x = 0
    if max_x > x_bound: max_x = x_bound
    return (min_x, min_y, max_x, max_y)

def calculate_mbr(bbxs, contours, ext, x_bound, y_bound):
    clusters, sources = find_clusters(bbxs, contours)
    mbrs = []
    for cluster, source in zip(clusters, sources):
        min_x = min(box[0] for box in cluster)
        min_y = min(box[1] for box in cluster)
        max_x = max(box[2] for box in cluster)
        max_y = max(box[3] for box in cluster)
        min_x, min_y, max_x, max_y = trim((min_x - ext, min_y - ext, max_x + ext, max_y + ext), x_bound, y_bound)

        mbrs.append(PotentialLeaking(min_x, min_y, max_x, max_y, source))
    return mbrs

# To be integrated
def compute_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def to_yolo_format(x, y, xw, yh, img_w, img_h):
    bw = xw - x
    bh = yh - y
    xc = x + bw / 2.0
    yc = y + bh / 2.0

    xc /= img_w
    bw /= img_w
    yc /= img_h
    bh /= img_h

    return xc, yc, bw, bh

def from_yolo_format(xc, yc, bw, bh, img_w, img_h):
    xc *= img_w
    bw *= img_w
    yc *= img_h
    bh *= img_h

    x, y = xc - bw / 2.0, yc - bh / 2.0
    xw, yh = x + bw, y + bh

    return list(map(int, (x, y, xw, yh)))

def heatmap_update(heatmap, bbox):
    # bbox for gain per reckoned bbox; mode -1 for loss per frame
    if bbox == -1:
        heatmap -= 10
    else:
        x, y, xw, yh = bbox
        heatmap[y:yh, x:xw] += 20
    heatmap[heatmap < 0] = 0
    heatmap[heatmap > 255] = 255
    return heatmap

def heatmap_eval(heatmap, bbox, thresh):
    x, y, xw, yh = bbox
    m = np.mean(heatmap[y:yh, x:xw])
    return m > thresh

def calculate_overlap_percentage(bbox, mask, use_contours=False):
    # YOLO bounding box coordinates (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = bbox.coords

    # Convert mask to binary (if necessary, assuming mask is already binary)
    mask_binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    contour_mask = np.zeros_like(mask_binary, dtype=np.uint8)
    if use_contours:
        contours = bbox.ancestors
        cv2.drawContours(contour_mask, contours, -1, (1), thickness=cv2.FILLED)
        intersection_mask = cv2.bitwise_and(contour_mask, mask_binary)
        intersection_area = np.sum(intersection_mask)
        roi = np.sum(contour_mask)
    else:
        intersection_area = np.sum(mask_binary[y_min:y_max, x_min:x_max])
        roi = (x_max - x_min) * (y_max - y_min)

    if roi == 0:
        return 0 

    overlap_percentage = (intersection_area / roi) * 100.0

    return (contour_mask * 255.0), overlap_percentage

def differentiate(prev_frame, current_frame):
    frame_delta_abs = cv2.absdiff(prev_frame, current_frame)
    frame_delta_win = calculate_ssim(cv2.blur(prev_frame, (3,3)), cv2.blur(current_frame, (3,3)))
    frame_delta_win *= 255.0
    frame_delta_win = frame_delta_win.astype("uint8")
    thresh_win = cv2.threshold(frame_delta_win, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh_win = cv2.erode(thresh_win, (5,5), iterations=3)
    frame_delta_combo = (255 - frame_delta_abs) // 2 + frame_delta_win // 2
    # frame_delta_combo = ((255 - frame_delta_abs) * 0.7 + frame_delta_win * 0.3).astype(np.uint8)
    # thresh = cv2.threshold(frame_delta_combo, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(255 - frame_delta_combo, 30, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(frame_delta_combo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1)
    thresh = cv2.dilate(thresh, (5,5), iterations=3)
    thresh = cv2.erode(thresh, (5,5), iterations=3)
    diff = 255 - frame_delta_combo
    return diff, thresh

def process_images_sequence(sequence, show=True, debug=False, use_model='', input_h=480, input_w=640, ignored_areas=[],
                            root_dir='./', image_id=0, clientId=0, facility=0,cameraNo=0,date='',
                            overlap_percentage_threshold=100, save_diff_map=True, save_visual_result=True, overlap_use_contours=True, retrieve_annos=False):

    if use_model:
        if "diff" in use_model:
            test_transforms = diff_transforms
            model = init_ResNetmodel(use_model)
        elif "mc" in use_model:
            test_transforms = mc_transforms
            model = init_ResNetmodel(use_model, mc=True)
        else:
            test_transforms = color_transforms
            model = init_ResNetmodel(use_model)

    # Saving paths initialization
    heatmap_dir = f"./heatmaps/{str(clientId)}/{str(facility)}/{str(cameraNo)}/"
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_path = os.path.join(heatmap_dir, "activation.png")

    if save_diff_map:
        diff_map_dir = sequence.replace(os.path.basename(sequence), "diff_maps") if retrieve_annos else f"{str(root_dir)}LeakAI/diff_maps/{str(clientId)}/{str(facility)}/{str(cameraNo)}/{str(date)}/"
        os.makedirs(diff_map_dir, exist_ok=True)

    if save_visual_result:
        visual_result_dir = f"./visual_results/{str(clientId)}/{str(facility)}/{str(cameraNo)}/{str(date)}"
        os.makedirs(visual_result_dir, exist_ok=True)

    # Input format determination
    levels, label = '', ''
    is_frame_pair = True
    site_level = 0
    if isinstance(sequence, list) and len(sequence) == 2:
        # If data is a two element tuple
        folder_path = sequence[0]
        all_paths = [sequence[:]]
        site_level = -4
        levels = sequence[0].split('/')[-4:-2]
        full_ID = "-".join(levels)
    elif retrieve_annos:
        folder_path = sequence
        all_paths = [sorted([os.path.join(folder_path, f) for f in os.listdir(os.path.join(folder_path)) if f.endswith((".jpg", ".jpeg", ",png"))])]
    else:
        folder_path = sequence
        base_dir = os.path.basename(folder_path)
        is_frame_pair = False
        if check_label_folder(base_dir):
            # If data is passed at label folder level
            all_paths = [sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ",png"))])]
            site_level = -4
            label = base_dir
        elif check_date_folder(base_dir):
            # If data is passed at date folder level
            site_level = -3
            if is_final_path(folder_path):
                # If data is not splitted into labels, i.e., date level is the final path
                all_paths = [sorted([os.path.join(folder_path, f) for f in os.listdir(os.path.join(folder_path)) if f.endswith((".jpg", ".jpeg", ",png"))])]
            else:
                # If data is further splitted into labels
                all_paths = [os.path.join(folder_path, 'positive', f)
                                        for f in os.listdir(os.path.join(folder_path, 'positive')) if f.endswith((".jpg", ".jpeg", ",png"))] + \
                            [os.path.join(folder_path, 'negative', f) 
                                        for f in os.listdir(os.path.join(folder_path, 'negative')) if f.endswith((".jpg", ".jpeg", ",png"))]
                all_paths = [sorted(all_paths, key=lambda x: os.path.basename(x))]
        levels = folder_path.split('/')[site_level:site_level+3]
        full_ID = "-".join(levels)

    # Mask initilization
    mask_path = '/'.join(folder_path.split('/')[:site_level+2]) + '/MASK.png'
    if not os.path.exists(mask_path):
        logging.warning(f"ImageID - {str(image_id)} - Mask file does not exist for facility - {facility} - camera - {cameraNo}")
        mask = None
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (input_w, input_h))
    mask_vis = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    if mask is None:
        print(f"Skipping mask assignment for facility {facility} and camera {cameraNo} as mask is None.")
    else:
        mask_vis[:, :, 1] = mask

    all_frames_for_gif = {}
    bboxes = []
    # The outermost loop handles situations that multiple folders were passed
    for image_files in all_paths:
        prev_frame = None
        detected_frames = 0
        undetected_frames = 0
        first_detected = -1 if label == 'positive' else 'N/A'
        frames_n = len(image_files)
        annos = defaultdict(list)
        # Main loop iterating frames
        for i, file_path in enumerate(image_files):
            filename = os.path.basename(file_path)
            frame = cv2.imread(file_path)

            if frame is None:
                continue

            img_h, img_w, img_c = frame.shape
            for ignored in ignored_areas:
                area_x, area_y = int(img_w*ignored[0]), int(img_h*ignored[1])
                area_w, area_h = int(img_w*ignored[2]), int(img_h*ignored[3])
                frame[area_y:area_y+area_h, area_x:area_x+area_w, :] = np.zeros((area_h, area_w, img_c))
            frame = cv2.resize(frame, (input_w, input_h))
            current_frame = preprocess_frame(frame)
            if (not is_frame_pair and i == 0) or not os.path.exists(heatmap_path):
                heatmap = np.zeros((input_h, input_w))
            else:
                heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
            heatmap = heatmap.astype(np.float32)

            contour_mask = np.zeros((input_h, input_w), dtype=np.uint8)
            
            if prev_frame is not None:
                # Frame differentiating
                diff, thresh = differentiate(prev_frame, current_frame)
                diff_vis = cv2.merge((diff, diff, diff))
                thresh_vis = cv2.merge((thresh, thresh, thresh))
                mc = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                mc[:, :, 3] = diff
                if save_diff_map:
                    cv2.imwrite(os.path.join(diff_map_dir, filename), diff_vis)
                    if not debug:
                        logging.info(f"ImageID - {str(image_id)} - successfully saved diff map - {str(filename)} to {str(diff_map_dir)}")

                # Contours extraction
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                detections = 0
                if use_model:
                    if "diff" in use_model: source = diff_vis
                    elif "mc" in use_model: source = mc
                    else: source = frame

                if not contours:
                    cv2.putText(frame, f"Detected: {detected_frames} Undetected: {undetected_frames}", (int(input_w*0.45), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"First Detected Frame: {first_detected}", (int(input_w*0.45), 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    all_frames_for_gif[filename] = cv2.cvtColor(np.hstack((frame, diff_vis)), cv2.COLOR_BGR2RGB)
                    continue
                
                # Extract largest connected components
                cache = defaultdict(int)
                for j, contour in enumerate(contours):
                    area = cv2.contourArea(contour)
                    if area < 10:
                        continue
                    cache[j] = area
                
                ranking = sorted(list(cache.keys()), key=lambda x: cache[x], reverse=True)
                cutoff = min(5, len(ranking))
                largest_area_ranking = ranking[:cutoff]
                box_candidates = []
                contour_candidates = []
                extension = 20

                for r in largest_area_ranking:
                    contour = contours[r]
                    contour_candidates.append(contour)
                    x, y, w, h = cv2.boundingRect(contours[r])
                    newx, newy, xw, yh = trim((x - extension, y - extension, x + w + extension, y + h + extension), input_w, input_h)
                    box_candidates.append((newx, newy, xw, yh))

                candidates = calculate_mbr(box_candidates, contour_candidates, 0, input_w, input_h)

                # Patch-wise prediction
                for candidate in candidates:
                    newx, newy, xw, yh = bbox = candidate.coords
                    centerx, centery, yolow, yoloh = yolo_coords = to_yolo_format(newx, newy, xw, yh, input_w, input_h)
                    annos[filename].append(f'0 {centerx} {centery} {yolow} {yoloh}')
                    if use_model:
                        cur_patch = source[newy:yh, newx:xw]
                        # adjusted_diff = transforms.functional.adjust_contrast(cv2_2_pil(cur_patch[:, :, 3]), 2)
                        # adjusted_diff = np.array(ImageOps.equalize(adjusted_diff))
                        # print(adjusted_diff.shape)
                        # normalized_diff = ((adjusted_diff - np.min(adjusted_diff)) / (np.max(adjusted_diff) - np.min(adjusted_diff)) * (255 - 50) + 50).astype(np.uint8)
                        # cur_patch[:, :, 3] = cv2.equalizeHist(cur_patch[:, :, 3])
                        # cur_patch[:, :, 3] = normalized_diff
                        # cur_patch[:, :, 3] = adjusted_diff
                        cur_patch[:, :, 3] = np.clip(cur_patch[:, :, 3] + 20, 0, 255)
                        cur_patch = cv2_2_pil(cur_patch)
                        transformed_patch = test_transforms(cur_patch)
                        output, _ = model(transformed_patch.unsqueeze(0))
                        y_prob = F.softmax(output, dim = -1)
                        is_normal = y_prob.argmax(1, keepdim = True)[0][0]
                        if not is_normal:
                            continue
                    
                    # Heatmap update
                    heatmap = heatmap_update(heatmap, bbox)
                    if heatmap_eval(heatmap, bbox, 30):
                        # check overlap percentage with mask
                        if mask is not None:
                            contour_mask, overlap_percentage = calculate_overlap_percentage(candidate, mask, use_contours=overlap_use_contours)
                            if not debug:
                                logging.info(f"ImageID - {str(image_id)} - found bbox overlap percentage {overlap_percentage}%")
                            box = LeakingBoundingBox(0 if float(overlap_percentage) >= float(overlap_percentage_threshold) else 1, *yolo_coords, 1.0)
                            detections += 0 if float(overlap_percentage) >= float(overlap_percentage_threshold) else 1
                        else:
                            box = LeakingBoundingBox(1, *yolo_coords, 1.0)
                            detections += 1
                        bboxes.append(box)
                        cv2.rectangle(frame, (newx, newy), (xw ,yh), (0, 255, 0) if box.class_label == 1 else (255, 255, 255), 2)
                        cv2.rectangle(diff_vis, (newx, newy), (xw ,yh), (0, 255, 0) if box.class_label == 1 else (255, 255, 255), 2)
                    else:
                        continue

                if detections:
                    detected_frames += 1
                    if first_detected == -1:
                        first_detected = i
                else:
                    undetected_frames += 1

                cv2.putText(frame, f"Detected: {detected_frames} Undetected: {undetected_frames}", (int(input_w*0.45), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"First Detected Frame: {first_detected}", (int(input_w*0.45), 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                vis_base = np.hstack((frame, thresh_vis))
                vis_overlay_mask = np.hstack((mask_vis, mask_vis))
                contour_vis = np.zeros((input_h, input_w, 3), dtype=np.uint8)
                contour_vis[:, :, 2] = contour_mask
                vis_overlay_contour = np.hstack((contour_vis, contour_vis))
                vis_overlay = cv2.addWeighted(vis_overlay_mask, 0.5, vis_overlay_contour, 0.5, 0)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vis = cv2.addWeighted(vis_base, 0.5, vis_overlay, 0.5, 0)
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                all_frames_for_gif[filename] = vis_rgb
                if save_visual_result:
                    cv2.imwrite(os.path.join(visual_result_dir, filename), vis_rgb)

                if show:
                    cv2.imshow('Detected Changes', vis)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        break

                heatmap = heatmap_update(heatmap, -1)

            # save heatmap
            cv2.imwrite(heatmap_path, heatmap.astype(np.uint8))
            prev_frame = current_frame
        
        # In case of running stage 1 code solely
        if retrieve_annos:
            return annos

        logging.info(f"ImageID - {str(image_id)} - Source: {full_ID} | Label: {label if label else 'full'} | Clip length: {frames_n} | Detected: {detected_frames} | Undetected: {undetected_frames} | First Detected: {first_detected} | Trigger Rate: {detected_frames / frames_n}")

    if show:
        cv2.destroyAllWindows()
    return all_frames_for_gif, bboxes
