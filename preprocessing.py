import numpy as np
import cv2
import image_manipulation
from multiprocessing import Pool
from p_tqdm import p_map
from itertools import repeat
from os import path, mkdir
import json
import glob
from math import ceil, floor
import shutil

class preprocessing:

    # Constructor
    def __init__(self, input_path: list, output_path, num_of_processes = 8):
        self.input_path = input_path
        self.output_path = output_path
        self.num_of_processes = num_of_processes
        try:
            params = json.load(open(output_path + '/parameters.json'))
            
            self.numpy_seed = params['numpy_seed']
            np.random.seed(self.numpy_seed)

            self.training_set_size = params['training_set_size']
            self.validation_set_size = params['validation_set_size']
            self.testing_set_size = params['testing_set_size']

            self.image_size = params['image_size']
            self.mixed_data = params['mixed_data']

            if self.mixed_data:
                self.general_parameters = params['general_parameters']
            else:
                self.training_parameters = params['training_parameters']
                self.validation_parameters = params['validation_parameters']
                self.testing_parameters = params['testing_parameters']
                
        except:
            raise Exception("No parameters.json file found in the output directory.")
    
    # Creates a directory if it does not exist
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

        print("Shuffling Data...")
        np.random.shuffle(self.input_path)

        if self.mixed_data:
            self.process_mixed_data()
        else:
            # Split the dataset first
            # Process the images
            pass

    def process_mixed_data(self):
        # Process all images first
            print("Processing and Augmenting images...")
            with Pool(self.num_of_processes) as p:
                p_map(self.process_image, repeat(f'{self.output_path}', len(self.input_path)),
                    self.input_path, repeat(self.general_parameters, len(self.input_path)))
            
            # Shuffle the dataset again
            self.input_path = glob.glob(f'{self.output_path}/*.png')
            np.random.shuffle(self.input_path)
            # Get the positives and negatives
            positives = [x for x in self.input_path if 'positive' in x]
            negatives = [x for x in self.input_path if 'negative' in x]

            print("Moving files to correct directory")
            # Put the images in the correct directories
            if ceil(len(self.input_path)*self.training_set_size) > len(negatives):
                raise Exception("Not enough negative images to fill training set")
            
            # Move images to train
            for i in range(ceil(len(self.input_path)*self.training_set_size)):
                # move the image to the training directory
                current_file_path = negatives.pop()
                new_file_path = current_file_path.replace(self.output_path, f'{self.output_path}/train')
                shutil.move(current_file_path, new_file_path)
            
            if ceil(len(self.input_path)*self.validation_set_size) > len(negatives):
                raise Exception("Not enough negative images to fill validation set")
            
            # Move images to valid
            for i in range(ceil(len(self.input_path)*self.validation_set_size)):
                current_file_path = negatives.pop()
                new_file_path = current_file_path.replace(self.output_path, f'{self.output_path}/valid')
                shutil.move(current_file_path, new_file_path)

            # Move images to test
            for i in range(len(positives)):
                current_file_path = positives.pop()
                new_file_path = current_file_path.replace(self.output_path, f'{self.output_path}/test')
                shutil.move(current_file_path, new_file_path)

            for i in range(len(negatives)):
                current_file_path = negatives.pop()
                new_file_path = current_file_path.replace(self.output_path, f'{self.output_path}/test')
                shutil.move(current_file_path, new_file_path)
            print("Finished preprocessing data")

    # Applies the data cleaning process to the images
    def apply_data_cleaning(self, image, parameters):
        if parameters['adaptive_histogram']:
            image = image_manipulation.adaptive_histogram(image)
        if parameters['watershed']:
            image = image_manipulation.watershed(image)

        image = image_manipulation.resize(image, size=self.image_size)
        return image

    # Applies the preprocessing to the image
    def process_image(self, output_path, image_path, parameters):
        image = cv2.imread(image_path)
        image = self.apply_data_cleaning(image, parameters)
        new_train_file_path = f"{output_path}/{image_path.replace('/','-').replace('.png', '')}"

        #augments the data
        new_images = image_manipulation.augment_data(image, hflip= parameters['hflip'], 
                                                     vflip=parameters['vflip'], 
                                                     max_rotation=parameters['max_rotation'])
        for index, img in enumerate(new_images):
            cv2.imwrite(f"{new_train_file_path}_{index}.png", img) 
        return