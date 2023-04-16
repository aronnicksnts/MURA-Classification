import MURA
import cv2
import image_manipulation
from multiprocessing import Pool
from p_tqdm import p_map
import itertools
import vae
from os import listdir
import numpy as np
import glob
from math import ceil, floor

# All editable variables
image_paths = MURA.MURA_DATASET()
new_file_path = "MURA-v1.1/augmented/test_1"
all_image_paths = image_paths.get_combined_image_paths().to_numpy()
# Numpy random seed for dataset shuffling
np.random.seed(15)
# Dataset Split - Should sum up to 1.0
training_set = 0.4
validation_set = 0.05
testing_set = 0.55


# horizontal flip augmentation
augment_hflip = False
# vertical flip augmentation
augment_vflip = False
# 0 for no rotation
max_rotation = 0
# max threads to be used
num_processes = 8
# num of epochs
epochs = 1
# num of batch size
batch_size = 32

# Applies various data_cleaning methods chosen from data_cleaning.py
def apply_data_cleaning(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_manipulation.adaptive_histogram(image)
    image = image_manipulation.watershed(image)
    # image = image_manipulation.black_and_white(image)
    image = image_manipulation.resize(image)
    return image

# Iterate on image paths and apply data cleaning processes
# Saves images on new_file_path and gets images from image_paths
# image_paths should be a tuple containing (integer, string of path, label)
def process_images(new_file_path, image_path):

    image = cv2.imread(image_path[0])
    image = apply_data_cleaning(image)
    new_train_file_path = f"{new_file_path}/{image_path[0].replace('/','-').replace('.png', '')}"

    #augments the data
    new_images = image_manipulation.augment_data(image, hflip= augment_hflip, vflip=augment_vflip, max_rotation=max_rotation)
    for index, img in enumerate(new_images):
        cv2.imwrite(f"{new_train_file_path}_{index}.png", img) 
    return


if __name__ == "__main__":
    print("Shuffling Dataset")
    np.random.shuffle(all_image_paths)

    print("Splitting Dataset")
    total_images = len(all_image_paths)
    positives = []
    negatives = []
    for image_path in all_image_paths:
        if image_path[1] == 1.0:
            positives.append(image_path[0])
        else:
            negatives.append(image_path[0])

    training = []
    validation = []
    testing = []
    for i in range(ceil(total_images*training_set)):
        training.append(negatives.pop())
    for i in range(floor(total_images*validation_set)):
        validation.append(negatives.pop())
    for i in range(len(positives)):
        testing.append(positives.pop())
    for i in range(len(negatives)):
        testing.append(negatives.pop())

    np.random.shuffle(testing)

    # PreProcesses the images
    # print("Processing Images")
    # with Pool(num_processes) as p:
    #     p_map(process_images, itertools.repeat(new_file_path, len(all_image_paths)), all_image_paths)

    # Puts all images in a single array and converts them into a numpy array
    # all_images = []
    # for image_path in glob.glob(f'{new_file_path}/*.png'):
    #     all_images.append(cv2.imread(image_path))
    # all_images = np.array(all_images)
    
    # Get all image paths
    # Creates and trains the model
    # vanilla = vae.Autoencoder()
    # vanilla.compile_AE()
    # model = vanilla.fit_AE(all_images, all_images)