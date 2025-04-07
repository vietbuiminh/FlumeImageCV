from load_contours import load_contours
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation, disk
from skimage.draw import line
from scipy.ndimage import label
import json
import numpy as np
import cv2
import os

def extract_green_contour(image_bgr):
    # Convert to HSV to isolate red
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Green range in HSV
    lower_green = np.array([35, 70, 50])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    return green_mask

def connect_close_segments(binary_image, max_distance=15):
    skeleton = skeletonize(binary_image > 0)
    labeled, num_features = label(skeleton)

    segments = []
    for i in range(1, num_features + 1):
        coords = np.column_stack(np.where(labeled == i))
        if coords.shape[0] == 0:
            continue
        elif len(coords) >= 2:
            segments.append((coords[0], coords[-1]))

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            for pt1 in segments[i]:
                for pt2 in segments[j]:
                    dist = np.linalg.norm(pt1 - pt2)
                    if dist < max_distance:
                        rr, cc = line(pt1[0], pt1[1], pt2[0], pt2[1])
                        skeleton[rr, cc] = 1

    thickened = dilation(skeleton, disk(1))
    return (thickened * 255).astype(np.uint8)

contours = load_contours('data/13b 17a_ABBA060115c.json')
calib_image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/9a 13a/calib-Jan1806/calib03.JPG')
skipping_no = 1
cnt_len = len(contours)
# turn calib image into a black background
calib_image = np.zeros_like(calib_image)

# for i, c in enumerate(contours[::skipping_no]):
#     cv2.drawContours(calib_image, [c], -1, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)
#     cv2.putText(calib_image, f"#{i*skipping_no}", tuple(c[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)
print(calib_image.shape)
cv2.drawContours(calib_image, [contours[100]], -1, (0, 255, 0), 2)
cv2.drawContours(calib_image, [contours[101]], -1, (0, 255, 0), 2)
# cv2.drawContours(calib_image, [contours[49]], -1, (0, 255, 0), 2)

cv2.imshow("Contours", calib_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

green_mask = extract_green_contour(calib_image)
cv2.imshow("Green Mask", green_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

fixed_cnts = connect_close_segments(green_mask, max_distance=100)
cv2.imshow("Fixed Contours", fixed_cnts)
cv2.waitKey(0)
cv2.destroyAllWindows()