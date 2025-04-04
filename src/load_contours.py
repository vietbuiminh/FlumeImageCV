import matplotlib.pyplot as plt
import json
import numpy as np
import cv2
import os

def load_contours(file_path):
    

    with open(file_path, 'r') as f:
        contours = json.load(f)

    # Convert the contours back to numpy arrays
    contours = [np.array(cnt) for cnt in contours]
    
    return contours

#load example contours and plot it
# if __name__ == "__main__":
#     # Load contours from the JSON file
#     contours = load_contours('data/13b 17a_ABBA060115c.json')
#     # Reverse the order of contours
#     contours.reverse()

#     # Create a blank image to draw contours
#     # image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/13b 17a/ABBA060114b/OLYMPUS DIGITAL CAMERA0001.JPG')
#     image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/9a 13a/calib-Jan1806/calib03.JPG')

#     # Plot the contours with indices
#     cnt_len = len(contours)
#     skipping_no = 1
#     for i, c in enumerate(contours[::skipping_no]):
#         cv2.drawContours(image, [c], -1, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)
#         cv2.putText(image, f"#{i*skipping_no}", tuple(c[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)

#     # Display the image with contours
#     cv2.imshow("Contours", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#iterate through the data folder and read all the json file and save each of the contours after drawing them on calibration image
data_folder = 'data'
output_folder = 'data/output_images'
os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(data_folder):
    if file_name.endswith('.json'):
        file_path = os.path.join(data_folder, file_name)
        contours = load_contours(file_path)
        contours.reverse()

        image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/9a 13a/calib-Jan1806/calib03.JPG')
        cnt_len = len(contours)
        skipping_no = 1

        for i, c in enumerate(contours[::skipping_no]):
            cv2.drawContours(image, [c], -1, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)
            cv2.putText(image, f"#{i*skipping_no}", tuple(c[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)

        output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_contours.jpg")
        cv2.imwrite(output_path, image)