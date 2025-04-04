import matplotlib.pyplot as plt
import json
import numpy as np
import cv2

def load_contours(file_path):
    

    with open(file_path, 'r') as f:
        contours = json.load(f)

    # Convert the contours back to numpy arrays
    contours = [np.array(cnt) for cnt in contours]
    
    return contours

#load example contours and plot it
if __name__ == "__main__":
    # Load contours from the JSON file
    contours = load_contours('data/13b 17a_ABBA060114a.json')
    # Reverse the order of contours
    contours.reverse()

    # Create a blank image to draw contours
    # image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/13b 17a/ABBA060114b/OLYMPUS DIGITAL CAMERA0001.JPG')
    image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/9a 13a/calib-Jan1806/calib03.JPG')

    # Plot the contours with indices
    cnt_len = len(contours)
    skipping_no = 1
    for i, c in enumerate(contours[::skipping_no]):
        cv2.drawContours(image, [c], -1, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 2)
        cv2.putText(image, f"#{i*skipping_no}", tuple(c[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255-255/cnt_len * (i*skipping_no), 255/cnt_len * (i*skipping_no)), 1)

    # Display the image with contours
    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()