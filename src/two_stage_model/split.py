import os
import random
import shutil

def read_images_from_directory(directory_path):
    """
    Read all image file paths from a given directory.
    """
    return [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(('.png', '.jpg', '.jpeg'))]

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets.
    """
    # Ensure the ratios sum up to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum up to 1"

    def split_data(data):
        random.shuffle(data)
        train_size = int(len(data) * train_ratio)
        val_size = int(len(data) * val_ratio)
        return data[:train_size], data[train_size:train_size + val_size], data[train_size + val_size:]

    train_pos, val_pos, test_pos = split_data(dataset['positive'])
    train_neg, val_neg, test_neg = split_data(dataset['negative'])

    return {
        'train': {'positive': train_pos, 'negative': train_neg},
        'val': {'positive': val_pos, 'negative': val_neg},
        'test': {'positive': test_pos, 'negative': test_neg}
    }

def save_split_datasets(split_dataset, output_dir):
    """
    Save the split datasets by creating directories and copying image files.
    """
    for split in ['train', 'val', 'test']:
        for category in ['positive', 'negative']:
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for file_path in split_dataset[split][category]:
                shutil.copy(file_path, split_dir)

# Example usage:
positive_images_dir = 'LeakAI_diff_data/positive'
negative_images_dir = 'LeakAI_diff_data/negative'

# Read images
dataset = {
    'positive': read_images_from_directory(positive_images_dir),
    'negative': read_images_from_directory(negative_images_dir)
}

# Split dataset
split_data = split_dataset(dataset)

# Save split datasets
output_directory = 'tidy_all_diff_dataset'
save_split_datasets(split_data, output_directory)