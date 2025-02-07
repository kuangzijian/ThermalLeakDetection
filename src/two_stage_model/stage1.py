import cv2
import numpy as np
import sys
import os
# Get the parent directory of 'two_stage_model', which is 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the 'src' directory to sys.path
sys.path.append(project_root)
from two_stage_model.dynamic_analysis import process_images_sequence

root_dir = "/dataset/618-50-01-0019-01-3-2024-09-03"
for subdir in os.listdir(root_dir):
    if subdir == ".DS_Store": continue
    curdir = os.path.join(root_dir, subdir, "images")
    annos = process_images_sequence(curdir, show=False, save_diff_map=True, save_visual_result=False, retrieve_annos=True)
    label_dir = os.path.join(root_dir, subdir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    for filename, labels in annos.items():
        with open(os.path.join(label_dir, filename.replace("jpg", "txt")), 'w') as f:
            for line in labels:
                f.write(f"{line}\n")

    print(f"{curdir} Done")

# _ = process_images_from_folder("/dataset/LeakAI-simulation-data-splitted-flattened-woBounds-mergedBbx/Group1/575-70-01-0060-01-1-2023-05-26-positive/images", show=True)
