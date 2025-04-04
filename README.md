# Image Data Flume Experiment Extraction

These scripts are designed for extracting contours from images by Computer Vision (cv2) package and saving them for later use. It includes functionality for loading images, processing them to find contours, and saving those contours in a structured format.

![Contour edge process, for example in src/extract_cnts_box.py](examplerun.gif)

![Contour edge evolution, for example in src/load_contours.py](image1.png)

### All Runs
All flume experiment runs that I need from from Kim & Muto (2007) for delta evolution modeling.

![all run from data/output_images folder](all_runs.png)

## Project Structure
tbd

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Extracting Contours**: Use the `extract_cnts_box.py` script to load images and extract contours. You can specify the reference box for contour extraction.

2. **Saving Contours**:(not using yet) After extracting contours, use the `save_contours.py` script to save the contours to `data/contours.json`.

3. **Loading Contours**: Use the `load_contours.py` script to read the saved contours from `data/contours.json` for further processing or analy`sis.