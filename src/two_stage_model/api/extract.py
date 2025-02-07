from collections import defaultdict
from diff import calculate_ssim
import cv2

im1_rgb, im2_rgb = cv2.imread("../61850010038013850010038013850010038013850010038013231220110019136729.jpg"), cv2.imread("../61850010038013850010038013850010038013850010038013231220110519142488.jpg")
im1, im2 = cv2.cvtColor(im1_rgb, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2_rgb, cv2.COLOR_BGR2GRAY)
diff = calculate_ssim(im1, im2)
diff *= 255.0
diff = diff.astype("uint8")
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
thresh = cv2.erode(thresh, (5,5), iterations=3)
cv2.imwrite("diff.jpg", diff)
cv2.imwrite("thresh.jpg", thresh)

contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

memo = defaultdict(int)

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area:
        memo[i] = area

ranking = sorted(list(memo.keys()), key=lambda x: memo[x], reverse=True)

candidates = [contours[i] for i in ranking[:5]]

bbx = im2_rgb.copy()

for c in candidates:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(bbx, (x, y), (x + w, y + h), (36,255,12), 2)

cv2.imwrite("bbx.jpg", bbx)