import sys
import cv2
import numpy as np
import glob

from PIL import Image
from PIL import ImageFilter

import pyocr

# Params for Pre-Processing
# PARAMS TESTED {13, 0}, {11,2}
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 0

# Params for Plate Detection
SML_CTR_MIN_RATIO = 0.01
SML_CTR_MAX_RATIO = 0.8
PLATE_MAX_ASPECT_RATIO = 8
CTR_MIN_EXTENT_RATIO = 0.75


def preprocess(img, color=True):
    # Pre-processing of image as follows:
    # 1. Convert to Gray Scale
    # 2. Histogram Equalization
    # 3. Image Blur (Smoothing) using 1x1 kernel
    # 4. Apply Bilateral Filter
    if color:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    img_gray = cv2.equalizeHist(img_gray)
    img_gray = cv2.blur(img_gray, (3, 3))
    img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)

    return cv2.adaptiveThreshold(img_gray, 255.0,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 ADAPTIVE_THRESH_BLOCK_SIZE,
                                 ADAPTIVE_THRESH_WEIGHT)


def find_plate_rectangle(img):
    # This function finds plate in the image.
    # The idea is to find a rectangle within the image

    _, contours, _ = cv2.findContours(img.copy(),
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_TC89_L1)
    contours = [c for c in contours
                if cv2.contourArea(c) > img.size * SML_CTR_MIN_RATIO
                and cv2.contourArea(c) < img.size * SML_CTR_MAX_RATIO]
    ret = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        ## 0.02 is epsilon
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            area = cv2.contourArea(c)
            rect_area = w * h
            extent = float(area) / rect_area
            aspect_ratio = float(w) / h
            if extent > CTR_MIN_EXTENT_RATIO and aspect_ratio < PLATE_MAX_ASPECT_RATIO:
                ret.append(approx)
    return ret


def extract_plate_value(plateContour, img, plate_id, plate_detected_location, plate_location):

    if plateContour is not None:
        x, y, w, h = cv2.boundingRect(plateContour)
        crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (0, 0), fx=4, fy=4)
#         hsv = preprocess_ocr(crop)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('{0}/{1}.png'.format(plate_location, plate_id), hsv)
        im = Image.open('{0}/{1}.png'.format(plate_location, plate_id))
        im.filter(ImageFilter.SHARPEN)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('{0}/{1}.png'.format(plate_detected_location,
                                         plate_id),
                    img)
        return ocr_plate(im)


def preprocess_ocr(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edg = cv2.Canny(img, 0, 100)
    structuring_element = cv2.getStructuringElement(cv2.MORPH_DILATE, (6, 6))
    morfo = cv2.dilate(img_edg, structuring_element, iterations=1)
    return morfo


def ocr_plate(im):
    tool = pyocr.get_available_tools()[0]
    txt = tool.image_to_string(im)
    return txt


def run_aplr(input_location, preprocess_location, plate_detected_location, plate_location):
    for image in glob.glob("{0}/*.png".format(input_location)):
        img = cv2.imread(image)
        preprocessed_image = preprocess(img)
        idx = image.split("/")[-1].split(".")[0]
        cv2.imwrite("{0}/{1}.png".format(preprocess_location,
                                         idx), preprocessed_image)
        detected_plate = find_plate_rectangle(preprocessed_image)
        if detected_plate:
            ocr_text = extract_plate_value(detected_plate[-1],
                                           img, idx,
                                           plate_detected_location,
                                           plate_location)
            print(ocr_text)


def run_once(input_file):
    print("Procesing {0}".format(input_file))
    img = cv2.imread(input_file)
    preprocessed_image = preprocess(img)
    idx = input_file.split(".")[0]
    cv2.imwrite("preprocessed_{0}.png".format(idx), preprocessed_image)
    detected_plate = find_plate_rectangle(preprocessed_image)
    if detected_plate:
        ocr_text = extract_plate_value(detected_plate[-1],
                                       img, idx,
                                       "sample_plate_detected",
                                       "sample_plate")
        print("Plate is: {0}".format(ocr_text.split("\n")[4]))


if __name__ == '__main__':
    run_once(sys.argv[1])
    # run_aplr("input_images", "preprocessed_images",
    #          "plate_detected", "plates")
