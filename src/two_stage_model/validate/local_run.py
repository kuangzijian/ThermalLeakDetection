import logging
from two_stage_model.dynamic_analysis import process_images_sequence


folder_path = '../../dataset/real_leak/618/618-50-01-0038-01/4/2023-12-20/positive'
# folder_path = '../../dataset/real_leak/618/618-50-01-0038-01/3/2023-12-20/positive'

# folder_path = '../../dataset/real_leak/575/575-70-01-0054-01/1/2024-02-07/positive'
# folder_path = "../../dataset/real_leak/575/575-70-01-0048-01/1/2024-05-15/positive"
# folder_path = '../../dataset/600-50-01-0405-01/1/2024-04-12/positive'
# folder_path = "../../dataset/LeakAI dataset 2024-12-28/618/618-50-01-0032-01/3/2023-12-28/positive"
# folder_path = '../../dataset/618-50-01-0018-01/2/2024-04-26/positive'
# folder_path = '../../dataset/611-50-01-0024-01/1/2024-05-08/positive'
# folder_path = "../../dataset/2024-06-22"
# folder_path = "../../dataset/real_leak/575/575-70-01-0046-01/1/2024-06-28"
# folder_path = 'test'
model_root = "../models/tsm/"
weights = "Leakai_mc_v8.pt"
import time
start = time.time()
gif_frames, bboxes = process_images_sequence(folder_path, show=True, debug=False, use_model=model_root+weights, overlap_use_contours=True)
end = time.time()
logging.info(f"Time elapsed: {end-start}")