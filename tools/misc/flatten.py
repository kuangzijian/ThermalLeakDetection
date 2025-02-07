import os
import shutil

def flatten_folders_and_add_image_subdir(root_dir, tar_dir):
    for subdir, dirs, _ in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            if dir_name == "positive":
                current_dir_path = os.path.join(subdir, dir_name)
                relative_path = os.path.relpath(current_dir_path, root_dir)
                new_dir_name = relative_path.replace(os.sep, '-')
                new_dir_path = os.path.join(tar_dir, new_dir_name)

                images_dir_path = os.path.join(new_dir_path, 'images')
                os.makedirs(images_dir_path, exist_ok=True)
                print(current_dir_path)
                for file in os.listdir(current_dir_path):
                    print(current_dir_path)
                    shutil.move(os.path.join(current_dir_path, file), os.path.join(images_dir_path, file))

                print(f"Moved contents of '{current_dir_path}' to '{images_dir_path}'")

# Example usage:
root_directory = "../../dataset/LeakAI-simulation-data-flattened-splitted"
tar_directroy = "../../dataset/LeakAI-simulation-data-splitted-flattened-woBounds-mergedBbx"
flatten_folders_and_add_image_subdir(root_directory, tar_directroy)