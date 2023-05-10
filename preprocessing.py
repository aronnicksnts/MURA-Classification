import numpy as np
import pandas as pd
import cv2
import image_manipulation
from multiprocessing import Pool
from p_tqdm import p_map
from itertools import repeat
from os import path, mkdir
import json

class preprocessing:

    # Constructor
    def __init__(self, input_path: list, output_path, num_of_processes = 8):
        self.input_path = input_path
        self.output_path = output_path
        self.num_of_processes = num_of_processes
        try:
            params = json.load(open(output_path + '/parameters.json'))

            self.numpy_seed = params['numpy_seed']
            self.training_set_size = params['training_set_size']
            self.validation_set_size = params['validation_set_size']
            self.testing_set_size = params['testing_set_size']
            self.image_size = params['image_size']
            
            if params['mixed_data']:
                self.training_parameters = params['training_parameters']
                self.validation_parameters = params['validation_parameters']
                self.testing_parameters = params['testing_parameters']
            else:
                self.general_parameters = params['general_parameters']
            print(vars(self))
        except:
            raise Exception("No parameters.json file found in the output directory.")
    

    def create_dir(parent_dir, new_dir_name):
        if path.isdir(f'{parent_dir}/{new_dir_name}'):
            return
        else:
            mkdir(f'{parent_dir}/{new_dir_name}')

    # Starts the preprocessing of the images
    def start(self):
        print("Creating directories...")
        preprocessing.create_dir(self.output_path, "train")
        preprocessing.create_dir(self.output_path, "valid")
        preprocessing.create_dir(self.output_path, "test")
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