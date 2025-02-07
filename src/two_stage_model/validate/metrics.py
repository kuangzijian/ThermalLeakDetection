import logging
import os
import re
from datetime import datetime
import cv2
from two_stage_model.dynamic_analysis import process_images_sequence
from engine_utils import get_inference_parameters

def setup_file_logging(log_dir):
    logger = logging.getLogger()
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"metrics.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


# define parameters
input_h, input_w, ignored_areas = get_inference_parameters()
model_root = "../models/tsm/"
weights = "Leakai_mc_v8.pt"
current_datetime=datetime.now().strftime('%Y%m%d_%H%M%S')
overlap_percentage_threshold = 80
overlap_use_contours=True
output_dir = f'../evaluation_results/{weights}_{current_datetime}/'
setup_file_logging(output_dir)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
test_path = "../../dataset/real_leak"
check = lambda dir: bool(re.match(r"^\d{4}-\d{2}-\d{2}$", dir))
remark = "hr_mask"

logging.info(f'model_name: {weights}, overlap_use_contours: {overlap_use_contours}, '
             f'overlap_percentage_threshold: {overlap_percentage_threshold}, output_dir: {output_dir}')

for dir_path, dir_names, file_names in os.walk(test_path):
    dir_path = dir_path.replace('\\', '/')
    if check(os.path.basename(dir_path)):
        facility_id, cam_id, date = dir_path.split('/')[-3:]
        all_frames = {}
        for label in {"positive", "negative"}:
            test_set = os.path.join(dir_path, label).replace('\\', '/')
            print(f"Processing {test_set}...")  # Debugging statement
            frame_dict, _ = process_images_sequence(test_set, show=False, debug=True, input_h=input_h, input_w=input_w, ignored_areas=ignored_areas,
                                                    overlap_percentage_threshold=overlap_percentage_threshold, use_model=model_root+weights, overlap_use_contours=overlap_use_contours)
            all_frames.update(frame_dict)
            print(f"Processed frames for {label}: {len(frame_dict)}")  # Debugging statement

        if remark and all_frames:
            output_path = os.path.join(output_dir, f"{facility_id}-{cam_id}-{date}-{remark}.mp4").replace('\\', '/')
            out = cv2.VideoWriter(output_path, fourcc, 5, (input_w*2, input_h))
            for p in sorted(all_frames):
                cur = cv2.cvtColor(all_frames[p], cv2.COLOR_BGR2RGB)
                out.write(cur)
            out.release()
            cv2.destroyAllWindows()
            logging.info(f"Generated video: {output_path}")
            print(f"Generated video: {output_path}")  # Debugging statement
        else:
            logging.warning(f"No frames to write for {facility_id}-{cam_id}-{date}-{remark}.")
            print(f"No frames to write for {facility_id}-{cam_id}-{date}-{remark}.")  # Debugging statement
