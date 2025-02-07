import re
import os

def check_date_folder(base_dir):
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", base_dir))

def check_label_folder(base_dir):
    return base_dir in {"positive", "negative"}

def is_final_path(base_dir):
    for sub in ['positive', 'negative']:
        sub_dir = os.path.join(base_dir, sub)
        if os.path.isdir(sub_dir):
            return False
    return True