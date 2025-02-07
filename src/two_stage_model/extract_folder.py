from collections import defaultdict
from diff import calculate_ssim
import cv2
import glob
import os

raw_dir = "../dataset/LeakAI.v11-stretched-to-640x640.yolov5pytorch/train/images/"
diff_dir = "../dataset/LeakAI_v11yolo_diff_data/train/images/"
save_dir = "../dataset/v11_train_bbx/"
os.makedirs(save_dir, exist_ok=True)
raws = glob.glob(os.path.join(raw_dir, "*.jpg"))
diffs = glob.glob(os.path.join(diff_dir, "*.jpg"))
pairs = defaultdict(str)
j = 0
for i in range(len(diffs)):
    diff, raw = diffs[i].split("/")[-1], raws[j].split("/")[-1]
    while diff != raw:
        j += 1
        raw = raws[j].split("/")[-1]

    pairs[diffs[i]] = raws[i]
    
for img in diffs:
    img_name = img.split("/")[-1]
    diff = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, (5,5), iterations=3)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    memo = defaultdict(int)

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area:
            memo[i] = area

    ranking = sorted(list(memo.keys()), key=lambda x: memo[x], reverse=True)

    candidates = [contours[i] for i in ranking[:5]]
    raw = cv2.imread(pairs[img])
    bbx = raw.copy()

    for c in candidates:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(bbx, (x, y), (x + w, y + h), (36,255,12), 2)

    cv2.imwrite(os.path.join(save_dir, img_name), bbx)