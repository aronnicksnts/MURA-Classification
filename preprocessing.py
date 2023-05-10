import numpy as np
import pandas as pd
import cv2
import image_manipulation
from multiprocessing import Pool
from p_tqdm import p_map
from itertools import repeat

class preprocessing:

    # Constructor
    def __init__(self, input_path: list, output_path, num_of_processes = 8, 
                 hflip: bool = False, vflip: bool = False, 
                 max_rotation: int = 0, resize: tuple = (0,0), black_and_white: bool = False, 
                 adaptive_histogram: bool = False, watershed: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.num_of_processes = num_of_processes

        self.hflip = hflip
        self.vflip = vflip
        self.max_rotation = max_rotation
        self.resize = resize
        self.black_and_white = black_and_white
        self.adaptive_histogram = adaptive_histogram
        self.watershed = watershed
    
    # Starts the preprocessing of the images
    def start(self):
        with Pool(self.num_of_processes) as p:
            p_map(self.process_image, repeat(f'{self.output_path}', len(self.input_path)) ,
                  self.input_path)
    
    def apply_data_cleaning(self, image):
        if self.black_and_white:
            image = image_manipulation.black_and_white(image)
        if self.adaptive_histogram:
            image = image_manipulation.adaptive_histogram(image)
        if self.watershed:
            image = image_manipulation.watershed(image)
        if self.resize != (0,0):
            image = image_manipulation.resize(image)
        return image

    # Applies the preprocessing to the image
    def process_image(self, output_path, image_path):
        image = cv2.imread(image_path)
        image = self.apply_data_cleaning(image)
        new_train_file_path = f"{output_path}/{image_path.replace('/','-').replace('.png', '')}"

        #augments the data
        new_images = image_manipulation.augment_data(image, hflip= self.hflip, vflip=self.vflip, 
                                                     max_rotation=self.max_rotation)
        for index, img in enumerate(new_images):
            cv2.imwrite(f"{new_train_file_path}_{index}.png", img) 
        return