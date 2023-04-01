import get_data
import pandas as pd
import numpy as np
import cv2
import image_manipulation

image_paths = get_data.MURA_DATASET()


# Applies various data_cleaning methods chosen from data_cleaning.py
def apply_data_cleaning(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_manipulation.adaptive_histogram(image)
    image = image_manipulation.watershed(image)
    return image

# Iterate on image paths and apply data cleaning processes
# Saves images on new_file_path and gets images from image_paths
# image_paths should be a tuple containing (integer, string of path, label)
def process_images(new_file_path, image_paths):
    for image_path in image_paths:
        image = cv2.imread(image_path[1])

        image = apply_data_cleaning(image)
        new_train_file_path = f"{new_file_path}/{image_path[1].replace('/','_')}"
        cv2.imwrite(new_train_file_path, image)
        print(new_train_file_path)
        break
    print('image saved')

process_images("MURA-v1.1/augmented/train", image_paths.get_image_paths(train=True).itertuples())

