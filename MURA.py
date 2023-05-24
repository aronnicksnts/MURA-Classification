import pandas as pd
import numpy as np
import os
import glob

class MURA_DATASET:
    def __init__(self):
        self.train_image_paths = pd.read_csv(f"MURA-v1.1/train_image_paths.csv", header=None)
        self.valid_image_paths = pd.read_csv(f"MURA-v1.1/valid_image_paths.csv", header=None)
        # Include only the HUMERUS part
        self.train_image_paths = self.train_image_paths[self.train_image_paths[0].str.contains("HUMERUS")]
        self.valid_image_paths = self.valid_image_paths[self.valid_image_paths[0].str.contains("HUMERUS")]

        # Add labels to the image itself
        self.train_image_paths.loc[self.train_image_paths[0].str.contains("positive"), 'label'] = 1
        self.train_image_paths.loc[self.train_image_paths[0].str.contains("negative"), 'label'] = 0
        self.valid_image_paths.loc[self.valid_image_paths[0].str.contains("positive"), 'label'] = 1
        self.valid_image_paths.loc[self.valid_image_paths[0].str.contains("negative"), 'label'] = 0

        # Get all image filepath and labels
        self.modified_dataset_path = f"MURA-v1.1/modified_dataset"

        # Get every images in the folder and label each of the images either 1 or 0
        self.modified_dataset_paths = glob.glob(f"{self.modified_dataset_path}/*.png")
        # Change "\\" to "/"
        self.modified_dataset_paths = [i.replace("\\", "/") for i in self.modified_dataset_paths]
        self.modified_dataset_paths = pd.DataFrame(self.modified_dataset_paths)
        for i in range(len(self.modified_dataset_paths)):
            self.modified_dataset_paths.loc[i, 'label'] = 1 if "positive" in self.modified_dataset_paths.loc[i, 0] else 0


    def get_image_paths(self, train = False):
        return self.train_image_paths if train else self.valid_image_paths
    
    def get_combined_image_paths(self):
        return pd.concat([self.train_image_paths, self.valid_image_paths], ignore_index=True)
    
    def get_modified_dataset_paths(self):
        return self.modified_dataset_paths
