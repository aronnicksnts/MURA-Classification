import MURA
import json
import preprocessing
import cv2
import p_tqdm


# Dataset Path
dataset_file_path = "datasets/test_cara_1"

# Open parameters inside dataset_path
params = json.load(open(dataset_file_path + '/parameters.json'))

# Dataset Path
image_paths = MURA.MURA_DATASET()
# Get what type of dataset to use
if params['modified_dataset']:
    all_image_paths = image_paths.get_modified_dataset_paths()
else:
    all_image_paths = image_paths.get_combined_image_paths()

all_image_paths = all_image_paths.to_numpy()[:,0]

# print(all_image_paths)
preprocess = preprocessing.preprocessing(input_path = all_image_paths, output_path = dataset_file_path)
if __name__ == '__main__':
    preprocess.start()