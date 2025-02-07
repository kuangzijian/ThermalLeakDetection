import os
import shutil
import glob

root_dir = "../dataset/LeakAI-seg.v2i.png-mask-semantic/"
set_folders = ["train", "valid", "test"]
data_folders = ["before", "after", "masks"]

tar_dir = "../dataset/organized_diffseg_dataset"
for path in (paths := [os.path.join(tar_dir, folder, data) for folder in set_folders for data in data_folders]):
    os.makedirs(path, exist_ok=True)

for folder in set_folders:
    src_path = os.path.join(root_dir, folder)
    # im_paths = [glob.glob(os.path.join(src_path, ext)) for ext in ["*.jpg", "*.png"]]
    im_paths = glob.glob(os.path.join(src_path, "*.jpg"))
    diff_paths = glob.glob(os.path.join(src_path, "*.png"))
    im_paths, diff_paths = sorted(im_paths), sorted(diff_paths)
    for i, (im_path, diff_path) in enumerate(zip(im_paths, diff_paths)):
        im_name = im_path.split("/")[-1]
        cur_site = im_name.split("_")[0][:14]
        if i == 0 or cur_site != site_identifier:
            site_identifier = cur_site
        else:
            shutil.copy(diff_path, os.path.join(tar_dir, folder, "masks", im_name.replace(".jpg", ".png")))
            shutil.copy(im_path, os.path.join(tar_dir, folder, "after", im_name))
            shutil.copy(im_paths[i-1], os.path.join(tar_dir, folder, "before", im_name))