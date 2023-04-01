import get_data
import pandas as pd
import numpy as np
import cv2

image_paths = get_data.MURA_DATASET()


# Iterate on image paths and apply data cleaning processes
for image_path in image_paths.get_image_paths(train=True).itertuples():
