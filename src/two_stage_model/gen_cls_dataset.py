import os
import json
import random
import cv2
import glob

# annotation_file_path = '../dataset/LeakAI.v11-stretched-to-640x640.coco/{}/_annotations.coco.json'
# images_directory = '../dataset/LeakAI.v11-stretched-to-640x640.coco/{}/'
# positive_samples_directory = '../dataset/LeakAI_v11_patch_cls_data/{}/leakage'
# negative_samples_directory = '../dataset/LeakAI_v11_patch_cls_data/{}/normal'
extension_pixels = 20
negative_samples_per_image = 3
seed = 1234
random.seed(seed)

folders = ["test", "valid", "train"]

def does_overlap(test_box, boxes):
    x1, y1, width1, height1 = test_box
    for box in boxes:
        x2, y2, width2, height2 = box
        if not (x1 + width1 < x2 or x2 + width2 < x1 or y1 + height1 < y2 or y2 + height2 < y1):
            return True
    return False

for folder in folders:

    annotation_file_path = '../dataset/LeakAI.v11-stretched-to-640x640.coco/{}/_annotations.coco.json'.format(folder)
    images_directory = '../dataset/LeakAI.v11-stretched-to-640x640.coco/{}/'.format(folder)
    diff_directory = '../dataset/test/{}/images'.format(folder)
    positive_samples_directory = '../dataset/LeakAI_v11_cleansed_determ_patch_cls_data/{}/leakage'.format(folder)
    negative_samples_directory = '../dataset/LeakAI_v11_cleansed_determ_patch_cls_data/{}/normal'.format(folder)

    os.makedirs(positive_samples_directory, exist_ok=True)
    os.makedirs(negative_samples_directory, exist_ok=True)

    diffs = os.listdir(diff_directory)

    with open(annotation_file_path, 'r') as f:
        annotations = json.load(f)

    for img_info in annotations['images']:
        img_id = img_info['id']
        img_name = img_info['file_name']
        img_path = os.path.join(images_directory, img_name)
        if img_path.split("/")[-1].replace(".jpg", ".png") not in diffs:
            continue
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        annotated_boxes = [ann['bbox'] for ann in annotations['annotations'] if ann['image_id'] == img_id]
        positive_boxes = []

        for box in annotated_boxes:
            x, y, width, height = box
            x1 = max(0, int(x - extension_pixels))
            y1 = max(0, int(y - extension_pixels))
            x2 = min(img_width, int(x + width + extension_pixels))
            y2 = min(img_height, int(y + height + extension_pixels))

            cropped_img = img[y1:y2, x1:x2]

            # stats["h"].append(y2-y1)
            # stats["w"].append(x2-x1)

            save_path = os.path.join(positive_samples_directory, f"{img_name}_{y1}_{y2}_{x1}_{x2}_positive.png")
            cv2.imwrite(save_path, cropped_img)
            positive_boxes.append((x1, y1, x2 - x1, y2 - y1))

        for _ in range(negative_samples_per_image):
            while True:
                rand_width = random.randint(150, 250)
                rand_height = random.randint(120, 220)
                x1 = random.randint(0, img_width - rand_width)
                y1 = random.randint(0, img_height - rand_height)
                y2, x2 = y1+rand_height, x1+rand_width
                if not does_overlap((x1, y1, rand_width, rand_height), positive_boxes):
                    cropped_img = img[y1:y2, x1:x2]
                    save_path = os.path.join(negative_samples_directory, f"{img_name}_{y1}_{y2}_{x1}_{x2}_negative.png")
                    cv2.imwrite(save_path, cropped_img)
                    break

# from collections import Counter
# hs, ws = sorted(stats["h"]), sorted(stats["w"])
# print(sum(hs)/len(hs), hs[len(hs)//2], max(Counter(hs), key=Counter(hs).get))
# print(sum(ws)/len(ws), ws[len(ws)//2], max(Counter(ws), key=Counter(ws).get))