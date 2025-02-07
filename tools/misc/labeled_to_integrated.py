import os
import shutil
import cv2

# For Windows 
integrated_diff_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\integrated_data_20240808\diff"
integrated_thermal_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\integrated_data_20240808\thermal"

# For Linux/MacOS
# integrated_diff_dir = "../../../dataset/integrated_data_20240724_shadow/diff"
# integrated_thermal_dir = "../../../dataset/integrated_data_20240724_shadow/thermal"


# classes = {"Negative", "Positive"}
classes = {"Negative", "Positive"}

for d in [integrated_diff_dir, integrated_thermal_dir]:
    for c in classes:
        os.makedirs(os.path.join(d, c), exist_ok=True)

# For Windows
labeled_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\600-50-01-0405-01-2-2024-07-19"

# For Linux/MacOS
# labeled_dir = "../../../dataset/LeakAI-shadow-data-labeled"


diff_lib = {}

# for group in os.listdir(labeled_dir):
#     group_dir = os.path.join(labeled_dir, group)
#     if os.path.isdir(group_dir):
#         for site in os.listdir(group_dir):
#             if not site.startswith("."):
#                 cur_folder = os.path.join(group_dir, site)
#                 breakpoint()
#                 diff_lib = {imname[:-4]: cv2.imread(os.path.join(cur_folder, "diff_maps", imname)) for imname in os.listdir(os.path.join(cur_folder, "diff_maps"))}
#                 for lbl in classes:
#                     for patch in os.listdir(os.path.join(cur_folder, lbl)):
#                         if not patch.startswith("."):
#                             imname, *coords = patch[:-4].split("_")
#                             x, y, w, h = map(int, coords)
#                             diff_patch = diff_lib[imname][y:y+h, x:x+w]
#                             cv2.imwrite(os.path.join(integrated_diff_dir, lbl, patch), diff_patch)
#                             shutil.copy(os.path.join(cur_folder, lbl, patch), os.path.join(integrated_thermal_dir, lbl, patch))

diff_lib = {imname[:-4]: cv2.imread(os.path.join(labeled_dir, "diff_maps", imname)) for imname in os.listdir(os.path.join(labeled_dir, "diff_maps"))}
for lbl in classes:
    for patch in os.listdir(os.path.join(labeled_dir, lbl)):
        if not patch.startswith("."):
            imname, *coords = patch[:-4].split("_")
            x, y, w, h = map(int, coords)
            diff_patch = diff_lib[imname][y:y+h, x:x+w]
            cv2.imwrite(os.path.join(integrated_diff_dir, lbl, patch), diff_patch)
            shutil.copy(os.path.join(labeled_dir, lbl, patch), os.path.join(integrated_thermal_dir, lbl, patch))
