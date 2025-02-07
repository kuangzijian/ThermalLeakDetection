import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import glob

classes = ["leakage", "normal"]

def calculate_mean_std(folder_path):
    means, stds = [], []
    image_files = [file for folder in classes for file in glob.glob(os.path.join(folder_path, folder, "*.png"))]
    for image_path in tqdm(image_files):
        image = Image.open(image_path)
        image_array = np.array(image) / 255.0  # Normalize the image array to [0, 1]

        # Calculate mean and std for each channel
        means.append(np.mean(image_array, axis=(0, 1)))
        stds.append(np.std(image_array, axis=(0, 1)))

    # Calculate the overall mean and std for the folder
    overall_mean = np.mean(means, axis=0)
    overall_std = np.mean(stds, axis=0)

    return overall_mean, overall_std

# Example usage
folder_path = "../dataset/LeakAI_v11_diff_patch_cls_data/train"
mean, std = calculate_mean_std(folder_path)
print("Mean:", mean)
print("Std:", std)