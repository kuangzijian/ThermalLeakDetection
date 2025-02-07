import os
import shutil
import re
from typing import Optional


def rename_image_files(root_dir, copy: Optional[str]=None):
    """
    Main interface.

    Parameters:
    :param root_dir: Path of the root-directory in which image files need to be renamed
    :param copy: None or a path string. When a path is provided, renamed files will be copied to the specified directory, else overwritten
    """

    if copy is not None:
        shutil.copytree(root_dir, os.path.join(copy, f"{os.path.basename(root_dir)}_backup"), copy_function=shutil.copy2)

    image_count_list = []
    all_images = 0

    check_reach_date = lambda dir: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", dir))
    check_reach_label = lambda dir: dir in {"positive", "negative"}

    for dir_path, dirnames, filenames in os.walk(root_dir):
        base_dir = os.path.basename(dir_path)
        # Check if reach bottom of tree
        # If have reached date folder level and no sub directories present
        if check_reach_date(base_dir) and not dirnames:
            ignored_levels = 1
            print(f"{dir_path} has not been splitted, please process later")
        # If have reached label folder level
        elif check_reach_label(base_dir):
            ignored_levels = 2
        else: continue
        image_count = sum(1 for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')))
        all_images += image_count
        image_count_list.append((dir_path, image_count))

        if os.name == 'nt':
            # Windows
            all_sub_dirs = os.path.normpath(dir_path).split(os.path.sep)
        else:
            # Mac/ Linux
            all_sub_dirs = dir_path.split(os.sep)

        site_id = all_sub_dirs[-ignored_levels-2]  # Site-ID => 575-70-01-0019-01
        camera_id = all_sub_dirs[-ignored_levels-1]  # Camera-ID => 1

        print(f"Processing Site-Id: {site_id}, \
                Appending Camera-ID: {camera_id}, \
                Base folder: {base_dir}")

        for file in filenames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # Getting the filename without extension
                filename_without_extension = os.path.splitext(file)[0]

                # Trimming all whitespaces, removing dashes, dots, and the first 2 characters
                if " " in filename_without_extension:
                    modified_name = filename_without_extension.replace(" ", "").replace("-", "").replace(".", "")
                    modified_name = modified_name[2:]
                else:
                    # If the iterating folder is already processed, do nothing
                    modified_name = filename_without_extension[-18:]

                # Generate new filename using site_id + camera_id + modified_name
                new_name = f"{site_id}{camera_id}{modified_name}.jpg"
                new_name = new_name.replace('-', '')
                # Renaming the image file
                os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_name))

if __name__ == '__main__':
    root = "/dataset/618-50-01-0019-01-3-2024-09-03"
    rename_image_files(root, copy = "../../dataset")