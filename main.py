import get_data
import cv2
import image_manipulation
from multiprocessing import Pool
from p_tqdm import p_map
import itertools
import vae

# All editable variables
image_paths = get_data.MURA_DATASET()
new_file_path = "MURA-v1.1/augmented/train"
all_image_paths = image_paths.get_image_paths(train=True).to_numpy()
# horizontal flip augmentation
augment_hflip = True
# vertical flip augmentation
augment_vflip = True
# 0 for no rotation
max_rotation = 0
# max threads to be used
num_processes = 8

# Applies various data_cleaning methods chosen from data_cleaning.py
def apply_data_cleaning(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_manipulation.adaptive_histogram(image)
    image = image_manipulation.watershed(image)
    image = image_manipulation.black_and_white(image)
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
    return


# if __name__ == "__main__":
#     print("Processing Images")
#     with Pool(num_processes) as p:
#         p_map(process_images, itertools.repeat(new_file_path, len(all_image_paths)), all_image_paths)
process_images(new_file_path, all_image_paths[0])
first_image = cv2.imread("MURA-v1.1/augmented/train/MURA-v1.1-train-XR_HUMERUS-patient02695-study1_positive-image1_0.png")
# vanilla = vae.Autoencoder()
# vanilla.compile_AE()
# model = vanilla.fit_AE(first_image, first_image)