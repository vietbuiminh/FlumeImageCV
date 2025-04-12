import cv2
import numpy as np
import json
import os
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils
import pandas as pd
import matplotlib.pyplot as plt

METHODS = ['wide', 'tight', 'auto']

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

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

def dilate_erode(image, dilate_iter=1, erode_iter=1):
    # Dilate and erode the edges
    dilated = cv2.dilate(image, None, iterations=dilate_iter)
    eroded = cv2.erode(dilated, None, iterations=erode_iter)
    return eroded


#========ALL CUSTOMIZABLE PARAMETERS===========
## NOTE: The parameters below are set to the values used in the original code. You may need to adjust them based on your specific requirements.
speed = 1 #ms for showing the window
core_path = '/Volumes/Extreme SSD/Ongoing Project/flume_experiments/' # replace this with the actual path these data are stored within your local
flume_experiment = '9a 13a' # replace this with the actual flume experiment name
experiment = 'ABBA060110b' # replace this with the actual experiment name

dilation_iter = 1
erode_iter = 1
blur_kernel = (15, 15) # prime number
contrast_alpha = 3 #4 #2
contrast_beta = 0
gray_threshold_low = 210
gray_threshold_high = 255
selected_method = 'tight' # 'wide', 'tight', 'auto'

# if selected "auto" then set the sigma value
auto_canny_sigma = 0.55

#========end of customizable parameters=========

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
cv2.imshow(f"First & Last Image of {flume_experiment}_{experiment}", np.hstack([first_image, last_image]))
# cv2.imshow("First Image", first_image)
# cv2.imshow("Last Image", last_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

calib_image = cv2.imread(last_image_path)
calib_gray = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)

# make the calib_gray contrast higher
high_contrast = cv2.convertScaleAbs(calib_gray, alpha=contrast_alpha, beta=contrast_beta)

# make threshold filter
threshold_filter = cv2.threshold(high_contrast, gray_threshold_low, gray_threshold_high, cv2.THRESH_BINARY)[1]

# blur with kernel provided
gauss_blur = cv2.GaussianBlur(threshold_filter, blur_kernel, 0)

# calib_gray = dilate_erode(calib_gray, dilate_iter=1, erode_iter=1)
# Display the images in a 2x2 grid
top_row = np.hstack([calib_gray, high_contrast])
bottom_row = np.hstack([threshold_filter, gauss_blur])
grid = np.vstack([top_row, bottom_row])

cv2.imshow("Image Processing Steps (2x2 Grid)", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

calib_gray = gauss_blur

cv2.imshow("Image", calib_image)
cv2.setMouseCallback("Image", click_event, ref_box)
cv2.waitKey(0)
cv2.destroyAllWindows()

# created a masked_image as the copy of the calib_image
mask = np.zeros(calib_image.shape[:2], dtype=np.uint8)
if ref_box == []:
    print("No reference box coordinates provided. Exiting.")
    exit()
# else:
#     #testing the mask
#     cv2.fillPoly(mask, [np.array(ref_box)], 255) #applying the mask
#     calib_gray = cv2.bitwise_and(calib_gray, calib_gray, mask=mask)
#     cv2.imshow("Masked Image", calib_gray)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print("Reference box coordinates: ", ref_box)

# v = np.median(calib_gray)
# sigma = 1
# lower = int(max(0, (1.0 - sigma) * v))
# upper = int(min(255, (1.0 + sigma) * v))

# plot a histogram of calib_gray displaying the stacked values of gray
# hist = cv2.calcHist([calib_gray], [0], None, [256], [0, 256])
# plt.plot(hist)
# plt.title("Histogram of Gray Image")
# plt.xlabel("Pixel Value")
# plt.ylabel("Frequency")
# plt.xlim([0, 256])
# plt.show()

# choose your method of edge detection
wide = cv2.Canny(calib_gray, 10, 250)
tight = cv2.Canny(calib_gray, 150, 200)
auto = auto_canny(calib_gray, auto_canny_sigma)

cv2.imshow("Original", calib_gray)
cv2.imshow("Edges", np.hstack([wide, tight, auto]))
key = cv2.waitKey(0) & 0xFF
if key == 27:  # ESC key
    cv2.destroyAllWindows()
    exit()
else:
    cv2.destroyAllWindows()

cnts_inside_all = []

contours_df = pd.DataFrame(columns=["image_file", "contours"])

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.convertScaleAbs(image_gray, alpha=contrast_alpha, beta=contrast_beta)
    image_gray = cv2.threshold(image_gray, gray_threshold_low, gray_threshold_high, cv2.THRESH_BINARY)[1]
    # image_gray = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    image_gray = cv2.GaussianBlur(image_gray, blur_kernel, 0)  
    # image_gray = cv2.bitwise_and(image_gray, image_gray, mask=mask)

    # using case condition to choose the method of edge detection
    edged = cv2.Canny(image_gray, 10, 250) # wide method
    if selected_method == 'tight':
        edged = cv2.Canny(image_gray, 150, 200)
    elif selected_method == 'auto':
        edged = auto_canny(image_gray, auto_canny_sigma)
    else:
        raise ValueError("Invalid method. Choose from 'wide', 'tight', or 'auto'.")

    edged = dilate_erode(edged, dilate_iter=dilation_iter, erode_iter=erode_iter)

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

    # merge all the contours inside the reference box
    # if len(cnts_inside) > 1:
    #     # Sort contours to ensure they are concatenated in a clockwise manner
    #     sorted_cnts = sorted(cnts_inside, key=lambda c: cv2.contourArea(c), reverse=True)
    #     concatenated_contour = sorted_cnts[0]
    #     for next_contour in sorted_cnts[1:]:
    #         # Find the closest points between the last point of the current contour and the next contour
    #         distances = dist.cdist(concatenated_contour[:, 0, :], next_contour[:, 0, :])
    #         min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
    #         # Reorder the next contour to start from the closest point
    #         next_contour = np.roll(next_contour, -min_dist_idx[1], axis=0)
    #         # Concatenate the contours
    #         concatenated_contour = np.concatenate((concatenated_contour, next_contour))
    #     cnts_inside = [concatenated_contour]
    # elif len(cnts_inside) == 0:
    #     print(f"No contours found inside the reference box in {image_file}.")
    #     continue
    # else:
    #     cnts_inside = [cnts_inside[0]]
    cnts_inside_all.extend(cnts_inside)

    for i, c in enumerate(cnts_inside):
        cv2.drawContours(image, [c], -1, (0, 0, 255), 2)
        cv2.putText(image, f"#{i}", tuple(c[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Contours Inside Reference Box", image)
    key = cv2.waitKey(speed) & 0xFF
    if key == 27:  # ESC key
        break
    else:
        cv2.destroyAllWindows()
    new_row = pd.DataFrame([{"image_file": image_file, "contours": [c.tolist() for c in cnts_inside]}])
    contours_df = pd.concat([contours_df, new_row], ignore_index=True)
save_contours_to_file(cnts_inside_all, save_path)

# Save the contours_df to a CSV file
csv_path = os.path.splitext(save_path)[0] + ".csv"
contours_df.to_csv(csv_path)
print(f"Contours saved to {save_path}")
print(f"Contours DataFrame saved to {csv_path}")