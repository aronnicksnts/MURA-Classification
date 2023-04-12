import pandas
import torch
import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils import data

DATA_PATH = 'C:/Users/Kyle/PycharmProjects/sample_ae/MURA-v1.1/'


def read_data(src_sub_path, dest_sub_path):
    class_label = {'negative': 0, 'positive': 1}
    path = DATA_PATH + src_sub_path

    for patient in tqdm(os.listdir(path)):
        patient_path = path + patient + '/'
        for study in tqdm(os.listdir(patient_path)):
            study_path = patient_path + '/' + study + '/'
            xray_result = study.split('_')[1]
            result_id = class_label[xray_result]
            for image_file in tqdm(os.listdir(study_path)):
                src = os.path.normpath('{}{}'.format(study_path, image_file))
                dest = os.path.normpath('{}{}/{}'.format(DATA_PATH, dest_sub_path, result_id))
                cmd = 'copy {} {}\\{}_{}'.format(src, dest, patient, image_file)
                if not os.path.exists(dest):
                    os.system('mkdir {}'.format(dest))
                os.system(cmd)
                print(f'cmd: {cmd}')


class Xray(data.Dataset):
    def __init__(self, main_path, img_size=64, transform=None):
        super(Xray, self).__init__()
        self.transform = transform
        self.file_path = []
        self.labels = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x

        #        print(f'labels: {os.listdir(main_path)}')
        for label in os.listdir(main_path):
            if label not in ['0', '1']:
                continue
            for file_name in tqdm(os.listdir(main_path + '/' + label)):
                #                print(f'file_name: {file_name}')
                data = sitk.ReadImage(main_path + '/' + label + '/' + file_name)
                data = sitk.GetArrayFromImage(data).squeeze()
                img = Image.fromarray(data).convert('L').resize((img_size, img_size), resample=Image.BILINEAR)
                self.slices.append(img)
                self.labels.append(int(label))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.slices)


def get_xray_dataloader(bs, workers, dtype='train', img_size=64, dataset='rsna'):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    path = DATA_PATH

    path += dtype
    print(f'Path: {path}')

    dset = Xray(main_path=path, transform=transform, img_size=img_size)
    train_flag = True if dtype == 'train' else False
    dataloader = data.DataLoader(dset, bs, shuffle=train_flag,
                                 drop_last=train_flag, num_workers=workers, pin_memory=True)

    return dataloader


if __name__ == '__main__':
    #    read_data('train/XR_HUMERUS/', 'train/')
    read_data('valid/XR_HUMERUS/', 'test/')
