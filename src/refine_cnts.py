from load_contours import load_contours, load_contours_csv
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, dilation, disk
from skimage.draw import line
from scipy.ndimage import label
import json
import numpy as np
import cv2
import os
from tabulate import tabulate
import pandas as pd


# Define a new click_event function to get points from the calib_image
def click_event(image):
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Point selected: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 50, (0, 0, 255), 2)  # Draw a larger circle with no fill (border only)
            cv2.imshow("Select Points", image)

    cv2.imshow("Select Points", image)
    cv2.setMouseCallback("Select Points", mouse_callback)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points


def get_transformation_matrix(src_points, dst_points):
    """
    Calculate the transformation matrix using the provided source and destination points.
    """
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src_points, dst_points)

# Function to calculate real-world distance between two points
def calculate_calibrated_distance(pt1, pt2, transform_matrix):
    pt1_transformed = cv2.perspectiveTransform(np.array([[pt1]], dtype=np.float32), transform_matrix)[0][0]
    pt2_transformed = cv2.perspectiveTransform(np.array([[pt2]], dtype=np.float32), transform_matrix)[0][0]
    distance = np.linalg.norm(pt1_transformed - pt2_transformed)
    return distance

def midpoint(pt1, pt2):
    return (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))

def extreme_points(c):
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return extLeft, extRight, extTop, extBot

def calculate_angle(pt1, pt2, pt3, transform_matrix):
    # Transform the points using the transformation matrix
    pt1_transformed = cv2.perspectiveTransform(np.array([[pt1]], dtype=np.float32), transform_matrix)[0][0]
    pt2_transformed = cv2.perspectiveTransform(np.array([[pt2]], dtype=np.float32), transform_matrix)[0][0]
    pt3_transformed = cv2.perspectiveTransform(np.array([[pt3]], dtype=np.float32), transform_matrix)[0][0]

    # Calculate the vectors
    a = np.array(pt1_transformed)
    b = np.array(pt2_transformed)
    c = np.array(pt3_transformed)
    ba = a - b
    bc = c - b

    # Calculate the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

calib_image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/9a 13a/calib-Jan1806/calib03.JPG')
csv_file_name = '13b 17a_ABBA060115c.csv'
df = load_contours_csv(f'data/{csv_file_name}')
# skipping_no = 1
cnt_len = len(df['contours'])
thickness = 4
speed=1

# Use the click_event function to find the 4 reference points
print("Please click on 4 points in the image to define the reference points.")
reference_points = click_event(calib_image)
if len(reference_points) != 4:
    print('Error: Exactly 4 points are required. Please rerun the program and select 4 points.')
    exit()
print(f"Selected reference points: {reference_points}")

# Define the real-world distances (in cm) between the points
calibration_distance_points = [
    (0, 0),      # Point 1
    (70, 0),     # Point 2
    (70, 60),    # Point 3
    (0, 60)      # Point 4
]


# Compute the transformation matrix
transformation_matrix = get_transformation_matrix(reference_points, calibration_distance_points)

# Example usage: Calculate distance between two points in the image
# point_a = (150, 250)  # Example point A in the image
# point_b = (250, 350)  # Example point B in the image
# distance = calculate_real_world_distance(point_a, point_b, transformation_matrix)
# print(f"Real-world distance between points: {distance:.2f} cm")

# Use the click_event function to select points and draw lines between them
print("TEST: Please click on points in the image. Press 'ESC' to finish.")
calib_image = cv2.imread('/Volumes/Extreme SSD/Ongoing Project/flume_experiments/9a 13a/calib-Jan1806/calib03.JPG')

# while cv2.waitKey(0) != 27:
#     selected_points = click_event(calib_image)

#     # Draw lines between every two consecutive points and calculate distances
#     for i in range(len(selected_points) - 1):
#         pt1 = selected_points[i]
#         pt2 = selected_points[i + 1]

#         # Draw the line
#         cv2.line(calib_image, pt1, pt2, (0, 255, 0), 2)

#         # Calculate the real-world distance
#         distance = calculate_real_world_distance(pt1, pt2, transformation_matrix)

#         # Display the distance on the line
#         mid_point = midpoint(pt1, pt2)
#         cv2.putText(calib_image, f"{distance:.2f} cm", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
#     cv2.imshow("Contours", calib_image)
#     if cv2.waitKey(0) == 27:  # ESC key
#         break
#     cv2.destroyAllWindows()

L1_list = []
L2_list = []
L3_list = []
angle_list = []

for file_name,each_contour in zip(df['image_file'],df['np_contours']):
    # print(len(each))
    calib_image = np.zeros_like(calib_image)

    # Plotting out all the contour to get a comprehensive image of the shape outlines
    # This is processing the discontinuity from the raw extracted data
    if len(each_contour) == 0:
        cv2.destroyAllWindows()
        print("No contours found")
        L1_list.append(0.0)
        L2_list.append(0.0)
        L3_list.append(0.0)
        angle_list.append(0.0)
        continue
    elif len(each_contour) > 1:
        loose_ends_list = []
        for i, c in enumerate(each_contour):
            cv2.drawContours(calib_image, [c], -1, (255, 255, 255), thickness)
            # Collect loose ends of each contour
            loose_ends = [tuple(c[0][0]), tuple(c[-1][0])]
            loose_ends_list.extend(loose_ends)
            # cv2.circle(calib_image, loose_ends[0], 5, (0, 0, 255), -1)
            # cv2.circle(calib_image, loose_ends[1], 5, (0, 255, 0), -1)

        # Connect loose ends based on distance
        for i in range(len(loose_ends_list)):
            for j in range(i + 1, len(loose_ends_list)):
                pt1 = loose_ends_list[i]
                pt2 = loose_ends_list[j]
                dist = np.linalg.norm(np.array(pt1) - np.array(pt2))  # using linear algebra for finding the distance
                if dist < 100:  # Threshold distance
                    contour = np.array([pt1, pt2], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.drawContours(calib_image, [contour], -1, (255, 255, 255), thickness=thickness)
    else:
        cv2.drawContours(calib_image, [each_contour[0]], -1, (255, 255, 255), 4)

        

    # STREAMLINING THE CONTOURS HERE INTO 1 BIG CONTOUR
    # Calculate the convex hull of the contours
    if len(calib_image.shape) == 3 and calib_image.shape[2] == 3:  # Check if the image is in BGR format
        calib_image = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(calib_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Convert the grayscale image back to BGR

    calib_image = cv2.cvtColor(calib_image, cv2.COLOR_GRAY2BGR)
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    cv2.drawContours(calib_image, [hull], -1, (0, 0, 255), 2)  # Draw the convex hull in red

    # Finding the center point from the largest contour
    M = cv2.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
	# draw the center of the shape on the image
    cv2.circle(calib_image, (cX, cY), 7, (255, 255, 255), -1)
    # Draw vertical and horizontal lines through the center point
    height, width, _ = calib_image.shape
    # Convert line points to numpy array
    line_points_horizontal = np.array([(cX, 0), (cX, height)])
    line_points_vertical = np.array([(0, cY), (width, cY)])
    # Draw lines on the image
    cv2.line(calib_image, (cX, 0), (cX, height), (255, 255, 255), 2)  # Vertical line
    cv2.line(calib_image, (0, cY), (width, cY), (255, 255, 255), 2)   # Horizontal line
    # Find intersection points of the vertical and horizontal lines with the largest contour
    intersection_points_h = np.empty((0, 2), dtype=int)
    intersection_points_v = np.empty((0, 2), dtype=int)

    # for i in range(len(largest_contour)):
    #     x,y = largest_contour[i][0]
    #     x=int(x)
    #     y=int(y)
    #     dist1 = cv2.pointPolygonTest(line_points_horizontal, (x, y), True)
    #     dist2 = cv2.pointPolygonTest(line_points_vertical, (x, y), True)
    #     if dist1 == 0:
    #         intersection_points_h = np.append(intersection_points_h, [[x, y]], axis=0)
    #     if dist2 == 0:
    #         intersection_points_v = np.append(intersection_points_v, [[x, y]], axis=0)
    
    # determine the most extreme points along the largest_contour
    # left,right,top,bottom = extreme_points(largest_contour)
    # determind the most extreme points for hull convex
    left_hull,right_hull,top_hull,bottom_hull = extreme_points(hull)

    intersection_points_v = np.array([left_hull, right_hull])
    intersection_points_h = np.array([top_hull, bottom_hull])

    # Draw the intersection points on the image
    for point in intersection_points_h:
        cv2.circle(calib_image, point, 5, (0, 255, 0), -1)  # Green dots for intersection points
        cv2.circle(calib_image, point, 10, (0, 255, 0), 2)
    for point in intersection_points_v:
        cv2.circle(calib_image, point, 5, (255, 0, 0), -1)  # Blue dots for intersection points
        cv2.circle(calib_image, point, 10, (255, 0, 0), 2)

    cv2.line(calib_image, top_hull, bottom_hull, (0, 255, 0), 2)
    cv2.line(calib_image, top_hull, left_hull, (255, 0, 0), 2)
    cv2.line(calib_image, top_hull, right_hull, (255, 0, 0), 2)

    L1 = calculate_calibrated_distance(top_hull, bottom_hull, transformation_matrix)
    L2 = calculate_calibrated_distance(top_hull, left_hull, transformation_matrix)
    L3 = calculate_calibrated_distance(top_hull, right_hull, transformation_matrix)
  
    # Calculate the angle at top_hull formed by left_hull and right_hull 
    # THIS ANGLE ACCOUNTED FOR TRANSFORMATION MATRIX
    angle = calculate_angle(left_hull, top_hull, right_hull, transformation_matrix) # assumming the smallest angle is the one we want
    # Draw a circular arc to represent the angle at top_hull
    # Draw a circular arc to represent the angle at top_hull
    radius = 25  # Radius of the arc

    # this is for the plotting not the actual angle
    start_angle = int(np.degrees(np.arctan2(left_hull[1] - top_hull[1], left_hull[0] - top_hull[0])))
    end_angle = int(np.degrees(np.arctan2(right_hull[1] - top_hull[1], right_hull[0] - top_hull[0])))

    # Ensure the angles are in the correct range to draw the smaller angle
    if end_angle > start_angle:
        start_angle, end_angle = end_angle, start_angle

    # Calculate the smaller angle
    if start_angle - end_angle > 180:
        end_angle += 360

    # Draw the arc for the smaller angle
    cv2.ellipse(calib_image, top_hull, (radius, radius), 0, end_angle, start_angle, (0, 255, 255), 2)

    # Display the angle value near the arc
    angle_text_position = (top_hull[0] + radius + 10, top_hull[1] - radius - 10)
    cv2.putText(calib_image, f"{angle:.2f} deg", angle_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(calib_image, f"{L1:.2f} cm", midpoint(top_hull,bottom_hull), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    left_midpoint = midpoint(top_hull, left_hull)
    cv2.putText(calib_image, f"{L1:.2f} cm", (left_midpoint[0] - 100, left_midpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(calib_image, f"{L3:.2f} cm", midpoint(top_hull,right_hull), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the file name on the image
    cv2.putText(calib_image, file_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Append the lengths and angle to the lists
    L1_list.append(L1)
    L2_list.append(L2)
    L3_list.append(L3)
    angle_list.append(angle)

    # Show everything
    cv2.imshow("Contours", calib_image)
    key = cv2.waitKey(speed)
    if key == 27:  # ESC key
        cv2.destroyAllWindows()
        break
    else:
        cv2.destroyAllWindows()

# Ensure the lists are reshaped to match the DataFrame's column structure
df['L1'] = pd.Series(L1_list)
df['L2'] = pd.Series(L2_list)
df['L3'] = pd.Series(L3_list)
df['angle'] = pd.Series(angle_list)
# Drop the 'np_contours' column from the DataFrame
df = df.drop(columns=['np_contours','contours'])
# Save the updated DataFrame to a new CSV file
output_csv_path = f'data/msred_{csv_file_name}'
df.to_csv(output_csv_path, index=False)
# print(f"Updated DataFrame saved to {output_csv_path}")
# Print the DataFrame in a tabular format
# print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))