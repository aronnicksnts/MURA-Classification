import pandas as pd
import numpy as np

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

    def get_image_paths(self, train = False):
        return self.train_image_paths if train else self.valid_image_paths
    
    def get_combined_image_paths(self):
        return pd.concat([self.train_image_paths, self.valid_image_paths], ignore_index=True)
