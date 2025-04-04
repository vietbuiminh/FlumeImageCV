import json

def save_contours(cnts_inside, file_path):
    contours_list = [contour.tolist() for contour in cnts_inside]
    with open(file_path, 'w') as json_file:
        json.dump(contours_list, json_file)