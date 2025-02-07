import cv2
import os
from skimage.metrics import structural_similarity as ssim

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
    score, diff = ssim(image1, image2, full=True)
    # print(f"SSIM: {score}")
    return diff

def calculate_ssim_dataset(src, tar, copy=False):
    os.makedirs(tar, exist_ok=True)
    print(src)
    for i, p in enumerate(sorted(glob.glob(src))):
        img_name = p.split("/")[-1]
        cur_site = img_name.split("_")[0][:site_identifier_len]
        if i == 0 or cur_site != site_identifier:
            site_identifier = cur_site
            pre = load_image(p)
        else:
            img = load_image(p)
            diff = calculate_ssim(img, pre)
            diff_name = img_name[:-4] + "_diff.jpg"
            diff *= 255.0
            pre = img
            save_images(os.path.join(tar, diff_name), diff)
            if copy:
                shutil.copy(p, os.path.join(tar, img_name))


if __name__ == "__main__":
    # train_path = "../LeakAI.v11-stretched-to-640x640.yolov5pytorch/train/images"
    # val_path = "../LeakAI.v11-stretched-to-640x640.yolov5pytorch/val/images"
    # test_path = "../LeakAI.v11-stretched-to-640x640.yolov5pytorch/test/images"
    src_dir = "../bincls_test_thermal_dataset_20231220/"

    save_dir = "bincls_test_dataset"

    site_identifier_len = 14

    # os.makedirs(save_dir, exist_ok=True)
    import glob
    import shutil

    # dests = ["train", "valid", "test"]
    dests = ["negative", "positive"]
    yolo = False

    for folder in dests:
        if yolo:
            calculate_ssim_dataset(os.path.join(os.path.join(src_dir, folder, "images"), "*.jpg"), os.path.join(save_dir, folder))
        else:
            calculate_ssim_dataset(os.path.join(os.path.join(src_dir, folder), "*.jpg"), os.path.join(save_dir, folder))

    # save_images("diff.jpg", calculate_ssim(load_image("img1.jpg"), load_image("img2.jpg")) * 255)

