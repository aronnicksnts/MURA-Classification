import cv2 as cv
import pandas as pd


def adaptive_histogram(image):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)