import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
import torch
import numpy as np
import random
random.seed(42)

class RSNADataset(Dataset):
    def __init__(self, csv_file, data_folder, split, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data_folder = data_folder
        self.transform = transform
        self.split = split
        # self.data = self.data[self.data['class'] != 'No Lung Opacity / Not Normal']

        # Filter the data based on the specified split
        self.data = self.data[self.data['split'] == split]
        # print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('trainnnnnnnnnnnnnn',self.data.iloc[idx, 8])
        image_folder = os.path.join(self.data_folder, 'train')
        # print('imageeeeeeeeeee', image_folder)
        image_path = os.path.join(image_folder, self.data.iloc[idx, 1] + '.jpg')
        # print(image_path)
        image = Image.open(image_path).convert('L')
        target = int(self.data.iloc[idx, 7])
        # print(target)
        if self.transform:
            image = self.transform(image)
        return image, target