import os
import csv
import shutil

def count_images_and_balance_folders(root_dir, group_n=3, tar_dir=None):
    if tar_dir is None: tar_dir = root_dir 
    subfolder_info = []

    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name, "images")
        if os.path.isdir(subdir_path):
            image_count = sum(1 for file in os.listdir(subdir_path) if file.lower().endswith(('.png', '.jpg', '.jpeg')))
            subfolder_info.append((subdir_name, image_count))

    subfolder_info.sort(key=lambda x: x[1], reverse=True)

    groups = { f'Group{i}': [] for i in range(1, group_n + 1) }
    counts = { f'Group{i}': 0 for i in range(1, group_n + 1) }

    for folder, count in subfolder_info:
        target_group = min(counts, key=counts.get)
        groups[target_group].append(folder)
        counts[target_group] += count

    for group, folders in groups.items():
        group_path = os.path.join(tar_dir, group)
        os.makedirs(group_path, exist_ok=True)
        for folder in folders:
            shutil.move(os.path.join(root_dir, folder), os.path.join(group_path, folder))

    with open(os.path.join(tar_dir, 'subfolder_image_counts.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subfolder Name', 'Number of Images', 'Assigned Group'])
        for group, folders in groups.items():
            for folder in folders:
                image_count = next((count for fname, count in subfolder_info if fname == folder), 0)
                writer.writerow([folder, image_count, group])

directory_path = "../../dataset/LeakAI-simulation-data-splitted-flattened-woBounds-mergedBbx"
# tar_path = "../../dataset/LeakAI-simulation-data-splitted-flattened-woBounds-mergedBbx"
count_images_and_balance_folders(directory_path, 4)
