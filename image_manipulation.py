import cv2 as cv
import pandas as pd
import numpy as np
import cv2
import imutils
import random

# Applies adaptive histogram to the image accepts accepts only grayscale
def adaptive_histogram(image):
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(grayimg)


# Applies the watershed method to the image accepts only grayscale images
# Currently outputs an RGB image with markers
def watershed(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv.watershed(image,markers)

    # Removes the Background from the image
    # image[unknown==255] = 0
    # Puts the Blue mark in the image
    # image[markers == -1] = [255,0,0]
    return image


def resize(image, size: tuple = (64,64)):
    return cv.resize(image, size)


def black_and_white(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Augments the data according to what is needed by the user
def augment_data(image, hflip: bool = False, vflip: bool = False, max_rotation: int = 0):
    images = [image]
    if hflip:
        images.append(cv2.flip(image, 1))
    if vflip:
        temp_images = []
        for image in images:
            temp_images.append(cv.flip(image, 0))
        images.extend(temp_images)
    if max_rotation:
        temp_images = []
        for image in images:
            temp_images.append(imutils.rotate(image, random.randint(-max_rotation, max_rotation)))
        images.extend(temp_images)
                               
    return images