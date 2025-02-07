import os
import shutil
import json


def merge_datasets(dataset_dir, config_file, output_dir):
    """
        Merge multiple labeled datasets into a dataset for direct train use.

        Args:
        dataset_dir (str): Dataset path where to-be-merged data is present.
        config_file (str): Configuration file storing label datasets to be used for the current version model.
        output_dir (str): Output directory name.
        shadow_alone (bool): Whether class "Shadow", if present, should be considered as an individual category or
            merged into "Negative".
        """
    with open(config_file, 'r') as file:
        config = json.load(file)
        data_dates = config['data_used']
        shadow_alone = bool(config['shadow_alone'])

    os.makedirs(output_dir, exist_ok=True)

    for date in data_dates:
        base_path = f'{dataset_dir}/{date}/{date}_splitted'
        for modality in ['diff', 'thermal']:
            for split in ['train', 'test', 'valid']:
                src_dir = os.path.join(base_path, modality, split)
                target_dir = os.path.join(output_dir, modality, split)
                if os.path.exists(src_dir):
                    for cat in os.listdir(src_dir):
                        s = os.path.join(src_dir, cat)
                        d = os.path.join(target_dir, cat)
                        if os.path.isdir(s):
                            shutil.copytree(s, d, dirs_exist_ok=True)
                        if cat == 'Shadow' and not shadow_alone:
                            merge_categories(d, d.replace(cat, "Negative"))


def merge_categories(src, dst):
    """
    Merge two categories.

    Args:
    src (str): Source directory path.
    dst (str): Destination directory path, the folder name to be maintained.
    """
    for f in os.listdir(src):
        src_path = os.path.join(src, f)
        dst_path = os.path.join(dst, f)
        shutil.copy2(src_path, dst_path)
        print(src_path, dst_path)
    shutil.rmtree(src)


if __name__ == "__main__":
    dataset_dir = "../../../dataset/data"
    config_file = "Leakai_mc_v8.json"
    output_dir = "test_Dataset"
    merge_datasets(dataset_dir, config_file, output_dir)
