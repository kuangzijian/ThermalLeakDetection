import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tifffile as tiff


def to_seconds(time):
    seconds = int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:])
    return seconds

def is_valid_interval(time1, time2, intv=180):
    time1 = to_seconds(time1)
    time2 = to_seconds(time2)
    difference = abs(time1 - time2)
    
    return difference >= intv

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def visualize_images(diff):
    cv2.imshow("111", diff)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

def save_images(path, diff):
    cv2.imwrite(path, diff)

def calculate_difference(image1, image2):
    return cv2.absdiff(image1, image2)

def calculate_ssim(image1, image2, win=None):
    score, diff = ssim(image1, image2, win_size=win,full=True)
    # print(f"SSIM: {score}")
    return score, diff

def calculate_abs(image1, image2):
    diff = cv2.absdiff(image1, image2)
    # print(f"SSIM: {score}")
    return diff

def calculate_ssim_dataset(dir, tar, copy=False, yolo=False):
    scores = []
    if yolo:
        src = os.path.join(dir, "images", "*.jpg")
        src_labels = os.path.join(dir, "labels", "*.txt")
        tar = os.path.join(tar, "images")
        images = glob.glob(src)
        annos = glob.glob(src_labels)
    else:
        src_labels = os.path.join(dir, "*.jpg")
        images = glob.glob(src_labels)
    os.makedirs(tar, exist_ok=True)
    for i, p in enumerate(sorted(images)):

        base, img_name = p.split("/")[-2:]
        cur_site = img_name.split("_")[0][:site_identifier_len]
        if i == 0 or cur_site != site_identifier:
            site_identifier = cur_site
            pre = load_image(p)
        else:
            img = load_image(p)
            score, diff = calculate_ssim(img, pre)
            scores.append(score)
            diff_name = img_name[:-4] + "_diff.jpg"
            diff *= 255.0
            pre = img
            # if base == positive:
            if yolo:
                tar_labels = tar.replace("images", "labels")
                os.makedirs(tar_labels, exist_ok=True)
                shutil.copy(annos[i], os.path.join(tar_labels, img_name.replace(".jpg", ".txt")))
                save_images(os.path.join(tar, img_name), diff)
            else:
                save_images(os.path.join(tar, diff_name), diff)
            if copy:
                shutil.copy(p, os.path.join(tar, img_name))

    return scores

def calculate_2ch_dataset(dir, tar, copy=False, yolo=False):
    scores = []
    if yolo:
        src = os.path.join(dir, "images", "*.jpg")
        src_labels = os.path.join(dir, "labels", "*.txt")
        tar = os.path.join(tar, "images")
        images = glob.glob(src)
        annos = glob.glob(src_labels)
    else:
        src_labels = os.path.join(dir, "*.jpg")
        images = glob.glob(src_labels)
    os.makedirs(tar, exist_ok=True)
    for i, p in enumerate(sorted(images)):

        base, img_name = p.split("/")[-2:]
        cur_site = img_name.split("_")[0][:site_identifier_len]
        if i == 0 or cur_site != site_identifier:
            site_identifier = cur_site
            pre = load_image(p)
        else:
            img = load_image(p)
            score, diff_ssim = calculate_ssim(img, pre)
            diff_abs = calculate_abs(img, pre)
            diff_ssim *= 255.0
            diff_ssim = diff_ssim.astype("uint8")
            diff = (255 - diff_abs) // 2 + diff_ssim // 2
            synth = np.stack((img, diff), 2)

            # scores.append(score)
            synth_name = img_name[:-4] + "_2ch.tiff"
            pre = img
            # if base == positive:
            if yolo:
                tar_labels = tar.replace("images", "labels")
                os.makedirs(tar_labels, exist_ok=True)
                shutil.copy(annos[i], os.path.join(tar_labels, img_name.replace(".jpg", ".txt")))
                tiff.imwrite(os.path.join(tar, img_name.replace(".jpg", ".tiff")), synth)
            else:
                tiff.imwrite(os.path.join(tar, synth_name), synth)
            if copy:
                shutil.copy(p, os.path.join(tar, img_name))

def calculate_finer_dataset(dir, tar, copy=False, yolo=False):
    scores = []
    if yolo:
        src = os.path.join(dir, "images", "*.jpg")
        src_labels = os.path.join(dir, "labels", "*.txt")
        tar = os.path.join(tar, "images")
        images = glob.glob(src)
        annos = glob.glob(src_labels)
    else:
        src_labels = os.path.join(dir, "*.jpg")
        images = glob.glob(src_labels)
    os.makedirs(tar, exist_ok=True)
    for i, p in enumerate(sorted(images)):

        base, img_name = p.split("/")[-2:]
        cur_site = img_name.split("_")[0][:site_identifier_len]
        if i == 0 or cur_site != site_identifier:
            site_identifier = cur_site
            pre = load_image(p)
            pre_path = p
        else:
            # print(pre_path, p)
            if not is_valid_interval(pre_path.split("/")[-1][20:26], p.split("/")[-1][20:26], 60):
                continue
            img = load_image(p)
            score, diff_ssim = calculate_ssim(img, pre)
            diff_abs = calculate_abs(img, pre)
            diff_ssim *= 255.0
            diff_ssim[diff_ssim > 255] = 255
            diff_ssim[diff_ssim < 0] = 0
            diff_ssim = diff_ssim.astype("uint8")
            diff = (255 - diff_abs) // 2 + diff_ssim // 2
            synth = diff

            # scores.append(score)
            synth_name = img_name[:-4] + ".png"
            pre = img
            pre_path = p
            # if base == positive:
            if yolo:
                tar_labels = tar.replace("images", "labels")
                os.makedirs(tar_labels, exist_ok=True)
                shutil.copy(annos[i], os.path.join(tar_labels, img_name.replace(".jpg", ".txt")))
                cv2.imwrite(os.path.join(tar, img_name.replace(".jpg", ".png")), synth)
            else:
                cv2.imwrite(os.path.join(tar, synth_name), synth)
            if copy:
                shutil.copy(p, os.path.join(tar, img_name))

if __name__ == "__main__":
    # train_path = "../LeakAI.v11-stretched-to-640x640.yolov5pytorch/train/images"
    # val_path = "../LeakAI.v11-stretched-to-640x640.yolov5pytorch/val/images"
    # test_path = "../LeakAI.v11-stretched-to-640x640.yolov5pytorch/test/images"
    src_dir = "../../dataset/LeakAI.v11-stretched-to-640x640.yolov5pytorch/"

    save_dir = "../../dataset/test"

    site_identifier_len = 20

    # os.makedirs(save_dir, exist_ok=True)
    import glob
    import shutil

    dests = ["train", "valid", "test"]
    # dests = ["negative", "positive"]
    label_handle = True
    all_scores = []

    for folder in dests:
        calculate_finer_dataset(os.path.join(src_dir, folder), os.path.join(save_dir, folder), yolo=label_handle)
        # all_scores.append(scores)

    # save_images("diff.jpg", calculate_ssim(load_image("img1.jpg"), load_image("img2.jpg")) * 255)

