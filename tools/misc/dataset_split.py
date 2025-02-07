import random
import os
import shutil

# For Linux/MacOS
# thermal_dir = '../../../dataset/integrated_data_20240724_shadow/thermal/'
# diff_dir = '../../../dataset/integrated_data_20240724_shadow/diff/'
# tar_thermal_dir = '../../../dataset/LeakAI_shadow_data_20240725/thermal/'
# tar_diff_dir = '../../../dataset/LeakAI_shadow_data_20240725/diff/'

# For Windows
thermal_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\integrated_data_20240808\thermal\\"
diff_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\integrated_data_20240808\diff\\"
tar_thermal_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\20240808_splitted\thermal\\"
tar_diff_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\20240808_splitted\diff\\"
random.seed(1234)
def split_dataset(base_dir, tar_dir, train_ratio, valid_ratio, test_ratio, categories):
    os.makedirs(tar_dir, exist_ok=True)
    splits = ['train', 'valid', 'test']

    for split in splits:
        for category in categories:
            os.makedirs(os.path.join(tar_dir, split, category), exist_ok=True)
            os.makedirs(os.path.join(tar_dir.replace("thermal", "diff"), split, category), exist_ok=True)

    for category in categories:
        category_path = os.path.join(base_dir, category)
        all_files = sorted([f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f'Processing {category_path}')
        random.shuffle(all_files)
        total_files = len(all_files)
        train_end = int(total_files * train_ratio)
        valid_end = train_end + int(total_files * valid_ratio)
        def move_files(start_idx, end_idx, split_type):
            print(f'Moving files from {start_idx} to {end_idx} for {split_type} split...')
            for file_name in all_files[start_idx:end_idx]:
                shutil.copy(os.path.join(category_path, file_name),
                            os.path.join(tar_dir, split_type, category, file_name))
                shutil.copy(os.path.join(category_path.replace("thermal", "diff"), file_name),
                            os.path.join(tar_dir.replace("thermal", "diff"), split_type, category, file_name))

        move_files(0, train_end, 'train')
        move_files(train_end, valid_end, 'valid')
        move_files(valid_end, total_files, 'test')

split_dataset(thermal_dir, tar_thermal_dir, 0.8, 0.1, 0.1, categories=['Positive', 'Negative'])