import cv2
import numpy as np
import json
import os
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils

def click_event(event, x, y, flags, param):
    polygon_points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(calib_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", calib_image)

def save_contours_to_file(cnts_inside, file_path):
    contours_list = [c.tolist() for c in cnts_inside]
    with open(file_path, 'w') as f:
        json.dump(contours_list, f)

speed = 1 #ms for showing the window

core_path = '/Volumes/Extreme SSD/Ongoing Project/flume_experiments/' # replace this with the actual path these data are stored within your local
flume_experiment = '13b 17a' # replace this with the actual flume experiment name
experiment = 'ABBA060115c' # replace this with the actual experiment name

ref_box = []
path_im_lib = os.path.join(core_path, flume_experiment, experiment)
save_path = f'data/{flume_experiment}_{experiment}.json'

with open(save_path, 'w') as f:
    json.dump([], f)

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--path_im_lib", type=str, default=path_im_lib, help="Path to the image library")
ap.add_argument("-s", "--save_path", type=str, default=save_path, help="Path to save contours")
args = ap.parse_args()

path_im_lib = args.path_im_lib
save_path = args.save_path

folder_path = args.path_im_lib
image_files = sorted([f for f in os.listdir(folder_path) if not f.startswith("._")])
first_image_path = os.path.join(folder_path, image_files[0])
last_image_path = os.path.join(folder_path, image_files[-1])
first_image = cv2.imread(first_image_path)
last_image = cv2.imread(last_image_path)
cv2.imshow("First Image", first_image)
cv2.imshow("Last Image", last_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

calib_image = cv2.imread(last_image_path)
calib_gray = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)
calib_gray = cv2.GaussianBlur(calib_gray, (15, 15), 0)

cv2.imshow("Image", calib_image)
cv2.setMouseCallback("Image", click_event, ref_box)
cv2.waitKey(0)
cv2.destroyAllWindows()
if ref_box == []:
    print("No reference box coordinates provided. Exiting.")
    exit()
else:
    print("Reference box coordinates: ", ref_box)

v = np.median(calib_gray)
sigma = 1.2
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

cnts_inside_all = []

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_blur = cv2.GaussianBlur(image_gray, (15, 15), 0)  

    auto_canny = cv2.Canny(image_gray_blur, lower, upper)
    dilated = cv2.dilate(auto_canny, None, iterations=1)
    eroded = cv2.erode(dilated, None, iterations=1)
    edged = eroded

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts:
        (cnts, _) = contours.sort_contours(cnts)

    ref_box = np.array(ref_box).reshape((-1, 1, 2)).astype(np.int32)

    cnts_inside = []
    for c in cnts:
        inside_points = []
        for point in c:
            if cv2.pointPolygonTest(ref_box, (int(point[0][0]), int(point[0][1])), False) >= 0:
                inside_points.append(point)

        if inside_points:
            new_contour = np.array(inside_points).reshape((-1, 1, 2)).astype(np.int32)
            cnts_inside.append(new_contour)

    cnts_inside_all.extend(cnts_inside)

    for i, c in enumerate(cnts_inside):
        cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
        cv2.putText(image, f"#{i}", tuple(c[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("Contours Inside Reference Box", image)
    cv2.waitKey(speed)
    cv2.destroyAllWindows()

save_contours_to_file(cnts_inside_all, save_path)