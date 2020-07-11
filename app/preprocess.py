from skimage.filters import threshold_local
import cv2
from PIL import Image
from pathlib import Path
from natsort import natsorted, ns

import os
import numpy as np
import tempfile
import argparse
import matplotlib.pyplot as plt
import glob
import imutils 


def get_image(file_path):
    if file_path != None:
        image = cv2.imread(file_path)
        return image
    else:
        return "Not able to get image path"
"""
def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    # add a dir here to change directory
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename
"""

def resize(image):
    # load image, resize to height 1600 with original ratio reserved
    if image.shape[0] > 1600:
        height = 1600
        width = int(image.shape[1] * height / image.shape[0])
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def preprocess_mobile_image(image):
    # convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 100, 200)

    T = threshold_local(edged, 11, offset = 10, method = "gaussian")
    edged = (edged > T).astype("uint8") * 255

    output = Image.fromarray(edged)
    output.save("temp/mobile_output.jpg")
    return True

def sentences_segmentate(img):
    # Converting to Gray 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using Sobel Edge Detection to Generate Binary 
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # Two valued
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # Expansion and Corrosion
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # Expansion once to make the outline stand out
    dilation = cv2.dilate(binary, element2, iterations=1)

    # Corrode once, remove details
    erosion = cv2.erode(dilation, element1, iterations=1)

    # Expansion again to make the outline more visible
    dilation2 = cv2.dilate(erosion, element2, iterations=2)
    
    # Finding Outlines and Screening Text Areas
    region = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        cnt = contours[i]

        # Calculate contour area and screen out small areas
        area = cv2.contourArea(cnt)
        if (area < 1000):
            continue

        # Find the smallest rectangle
        rect = cv2.minAreaRect(cnt)
        #print("Rect: ", rect)

        # Box is the coordinate of four points
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Computing height and width
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # According to the characteristics of the text, select those too thin rectangles, leaving flat ones.
        if (height > width * 1.3):
            continue

        region.append(box)

    # Segmentate into images
    img_output_path = 'temp/out_sentences/'
    i = 0
    for box in region:
        x, y, w, h = cv2.boundingRect(box)
        ROI = img[y:y+h, x:x+w]

        path = os.path.join(img_output_path, '{}.jpg'.format(i))
        cv2.imwrite(path, ROI)
        i += 1

    cv2.drawContours(img, region, -1, (0, 255, 0), 3)
    #cv2.imwrite('C:\\Users\\linhn\\Desktop\\ML_final\\app\\testdata\\out_sentences\\%s\\result.jpg', img)

def image_resize_for_model(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
    
def resize_sentences_for_model(images):
    i = 0
    for image_path in images:
        img = cv2.imread(image_path)
        img = image_resize_for_model(img, height=32)
        
        color = [255, 255, 255] # 'cause purple!
        right = int(800 - img.shape[1]) * 1

        img_with_border = cv2.copyMakeBorder(img, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=color)

        output_path = "temp/out_sentences_resized/"
        output = Image.fromarray(img_with_border)
        output_path = os.path.join(output_path, '{}.jpg'.format(i))
        output.save(output_path)
        i += 1

def preprocess_for_model(images_path):
    # 1. Read image
    img = tf.io.read_file(image)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_jpeg(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])

    return img
